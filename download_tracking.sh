git clone --filter=blob:none --no-checkout --depth 1 --sparse https://github.com/ifzhang/ByteTrack.git && \
cd ByteTrack && \
git sparse-checkout set yolox && \
git checkout && \
cd ..
cp -r ByteTrack/yolox yolox
cp container-batch-inference/byte_tracker.py yolox/tracker/
sudo rm -r ByteTrack
