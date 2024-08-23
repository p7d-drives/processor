from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from io import IOBase

from tqdm import tqdm


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator


from typing import List
import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


from ultralytics import YOLO
import torch
model = YOLO("yolov8x.pt")
if torch.cuda.is_available():
  print('Cuda found available')
  model.to('cuda')
else:
  print('No CUDA found')
model.fuse()

CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2, 3, 5, 7]


class CustomCounter:
  def __init__(self, zones: list):
    from collections import defaultdict
    self.current_zone = defaultdict(int)
    self.last_zone = defaultdict(int)
    self.count = defaultdict(lambda: defaultdict(int))
    self.zones = zones

  def move(self, car_id, class_id, to_zone):
    if self.current_zone[car_id] == to_zone:
      return
    from_zone = self.current_zone[car_id] or self.last_zone[car_id]
    self.count[from_zone, to_zone][CLASS_NAMES_DICT[class_id]] += 1
    if to_zone == 0:
      self.last_zone[car_id] = self.current_zone[car_id]
    self.current_zone[car_id] = to_zone

  def get_zone(self, xyxy):
    pts = [np.zeros((2,)) for i in range(4)]
    pts[0][:] = (xyxy[0], xyxy[1])
    pts[1][:] = (xyxy[0], xyxy[3])
    pts[2][:] = (xyxy[2], xyxy[1])
    pts[3][:] = (xyxy[2], xyxy[3])
    for i, zone in enumerate(self.zones, 1):
      # zone => (ndarray((2,)) start, finish, direction +-1)
      for p in pts:
        if np.cross(zone.finish - zone.start, p - zone.start) <= 0:
            break
      else:
        return i
    return 0

  def check_car(self, xyxy, confidence, class_id, tracker_id):
    zone_i = self.get_zone(xyxy)
    self.move(tracker_id, class_id, zone_i)

  def iter_detections(self, det):
    for sth in det:
      self.check_car(*sth)


class Zone:
    start: np.ndarray
    finish: np.ndarray
    color: tuple[int, int, int]

    def __init__(self, start, finish, color):
        self.start = np.array(start, dtype=int)
        self.finish = np.array(finish, dtype=int)
        self.color = tuple(reversed(color))  # RGB to BGR


class VideoProcessor:
    BASE_RESOLUTION = 1080

    def __init__(self, video_path: str, output_path: str, zones: list[Zone]):
        self.video_path = video_path
        self.output_path = output_path
        self.zones = zones
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.video_info = VideoInfo.from_video_path(video_path)
        self.generator = get_video_frames_generator(video_path)
        self.custom_counter = CustomCounter(zones)
        annot = self._get_cv_annotation_specs()
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=annot[0], text_thickness=annot[0], text_scale=annot[1])

    def draw_zone(self, frame, zone: Zone, label):
        thickness, fontscale = self._get_cv_zone_specs()
        x1y1, x2y2, lineColor = zone.start, zone.finish, zone.color
        vtext = f'v {label} v'
        import cv2
        mid = list(map(int, (x1y1 + x2y2) // 2))
        cv2.line(frame, x1y1, x2y2, lineColor, thickness)
        text_img = np.zeros_like(frame)
        angle = -np.degrees(np.arctan2((x2y2 - x1y1)[1], (x2y2 - x1y1)[0]))
        main_rotation_matrix = cv2.getRotationMatrix2D(mid, angle, 1)
        (_w, _h), _baseline = cv2.getTextSize(vtext, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
        # _h += baseline
        cv2.putText(text_img, vtext, (mid[0] - _w // 2, mid[1] - _h // 2), cv2.FONT_HERSHEY_SIMPLEX, fontscale, lineColor, thickness, cv2.LINE_AA)
        rotated_text_img = cv2.warpAffine(text_img, main_rotation_matrix, (text_img.shape[1], text_img.shape[0]))
        frame = cv2.add(frame, rotated_text_img)
        # cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness, cv2.LINE_AA)

        return frame


    def process_video_iter(self, use_tqdm=False):
        with VideoSink(self.output_path, self.video_info) as sink:
            # loop over video frames
            gen = self.generator if not use_tqdm else tqdm(self.generator, total=self.video_info.total_frames)
            for frame in gen:
                # model prediction on single frame and conversion to supervision Detections
                results = model(frame, verbose=False)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # filtering out detections with unwanted classes
                mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # tracking detections
                tracks = self.byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )
                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # format custom labels
                labels = [
                    f"@{self.custom_counter.get_zone(xyxy)} #{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    for xyxy, confidence, class_id, tracker_id
                    in detections
                ]
                # updating line counter
                self.custom_counter.iter_detections(detections)
                # annotate and display frame
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)

                for i, zone in enumerate(self.custom_counter.zones, 1):
                    frame = self.draw_zone(frame, zone, str(i))

                import cv2
                ystart = 25
                y = ystart

                CV_FONT_THICK, CV_FONT_SCALE = self._get_cv_stats_text_specs()

                (wmax, _), _ = cv2.getTextSize('Stats:', cv2.FONT_HERSHEY_SIMPLEX, CV_FONT_SCALE, CV_FONT_THICK)

                text = ['Stats:'] + [f'{key[0]}->{key[1]}: {", ".join(f"{kv[0]}={kv[1]}" for kv in val.items())}; total={sum(val.values())}'
                                     for key, val in self.custom_counter.count.items() if 0 not in key]

                for textline in text:
                    (_w, dh), _ = cv2.getTextSize(textline, cv2.FONT_HERSHEY_SIMPLEX, CV_FONT_SCALE, CV_FONT_THICK)
                    wmax = max(wmax, _w)
                    y += dh + 5

                sub = frame[0:y+ystart, 0:wmax+2*ystart]
                overlay = np.zeros(sub.shape, dtype=np.uint8)
                frame[0:y+ystart, 0:wmax+2*ystart] = cv2.addWeighted(sub, 0.5, overlay, 0.5, 1.0)

                y = ystart
                for textline in text:
                    (_w, dh), _ = cv2.getTextSize(textline, cv2.FONT_HERSHEY_SIMPLEX, CV_FONT_SCALE, CV_FONT_THICK)
                    y += dh + 5
                    frame = cv2.putText(frame, textline, (ystart, y), cv2.FONT_HERSHEY_SIMPLEX, CV_FONT_SCALE, (255, 255, 255), CV_FONT_THICK, cv2.LINE_AA)

                sink.write_frame(frame)
                yield self.custom_counter.count

    def process_video(self, use_tqdm=False):
        for it in self.process_video_iter(use_tqdm):
           pass

    def process_video_logger(self, outfile_json, frequency=None, use_tqdm=False):
        import os
        if not frequency:
           frequency = 30
        json_last = {'frames': 0, 'fps': self.video_info.fps, 'total_frames': self.video_info.total_frames, 'chunks': []}
        for i, it in enumerate(self.process_video_iter(use_tqdm)):
            json_last['frames'] = i
            from collections import defaultdict
            to_json = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for (zf, zt), tsdict in it.items():
                if zf == 0 or zt == 0:
                   continue
                for ts, cnt in tsdict.items():
                    to_json[str(zf)][str(zt)][ts] = cnt
            json_last['chunks'].append(to_json)
            if (i + 1) % frequency == 0:
                with open(outfile_json + '.1', 'w') as fp:
                    import json
                    json.dump(json_last, fp)
                os.rename(outfile_json + '.1', outfile_json)
        os.system(f'ffmpeg -i {self.output_path} -c:v libx264 {self.output_path}.1.mp4')
        print('OK ffmpeg')
        os.rename(self.output_path + '.1.mp4', self.output_path)
        json_last['frames'] = self.video_info.total_frames
        with open(outfile_json + '.1', 'w') as fp:
            import json
            json.dump(json_last, fp)
        os.rename(outfile_json + '.1', outfile_json)


    def _get_cv_annotation_specs(self):  # thickness & fontscale
       return 1 * self.video_info.height // self.BASE_RESOLUTION, 0.5 * self.video_info.height / self.BASE_RESOLUTION

    def _get_cv_stats_text_specs(self):  # thickness & fontscale
       return 1 * self.video_info.height // self.BASE_RESOLUTION, 0.75 * self.video_info.height / self.BASE_RESOLUTION

    def _get_cv_zone_specs(self):  # thickness & fontscale
       return 2 * self.video_info.height // self.BASE_RESOLUTION, 1 * self.video_info.height / self.BASE_RESOLUTION


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='p7dtTrm ML part')
    parser.add_argument('-i', '--infile', dest='infile')
    parser.add_argument('-o', '--outfile', dest='outfile')
    parser.add_argument('-L', '--linesfile', dest='linesfile', required=False, default=None)
    parser.add_argument('-j', '--outjson', dest='outjson', required=False, default=None)
    parser.add_argument('--freq', dest='frequency', required=False, default=None)
    parser.add_argument('--no-tqdm', dest='no_tqdm', action='store_true')
    args = parser.parse_args()
    zones = [Zone((1231, 741), (1263, 319), (255, 128, 0)),
            Zone((69, 561), (670, 850), (0, 255, 0)),
            Zone((856, 18), (70, 625), (0, 128, 255)),
            Zone((1164, 273), (870, 228), (255, 0, 128))]

    if args.linesfile:
        zones.clear()
        # [{"start": [1, 2], "finish": [3, 4], "color": [5, 6, 7]}]
        import json
        with open(args.linesfile) as fp:
            zones_raw = json.load(fp)
            for zone in zones_raw:
                zones.append(Zone(zone['start'], zone['finish'], zone['color']))

    print([v for k, v in CLASS_NAMES_DICT.items() if k in CLASS_ID])
    vp = VideoProcessor(args.infile, args.outfile, zones=zones)
    if not args.outjson:
        vp.process_video(use_tqdm=not args.no_tqdm)
    else:
        vp.process_video_logger(args.outjson, args.frequency, use_tqdm=not args.no_tqdm)

