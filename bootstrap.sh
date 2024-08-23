#!/usr/bin/env sh
curdir="$PWD"
pip3 install -q ultralytics supervision==0.1.0
git clone https://github.com/ifzhang/ByteTrack.git

cd ByteTrack
#sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt
#sed -i 's/onnxruntime==1.8.0/onnxruntime==1.12.0/g' ./requirements.txt
find ./yolox -name '*.py' -exec 'sed' '-i' '-E' 's/np.float([^0-9]|$)/float\1/g' '{}' ';'
#pip3 install -q protobuf\<3.21
pip3 install -q -r ./requirements.txt
python3 setup.py -q develop
pip3 install -q cython_bbox onemetric loguru lap thop

cd "$curdir"

export PYTHONPATH="$PYTHONPATH:$curdir/ByteTrack"

python3 -c '__import__("ultralytics").checks()'
python3 -c 'print("yolox version:", __import__("yolox").__version__)'
python3 -c 'print("supervision version:", __import__("supervision").__version__)'

