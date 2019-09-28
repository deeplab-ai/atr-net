# Faster-RCNN
if [ ! -d "faster_rcnn" ]; then
    mkdir faster_rcnn
    cd faster_rcnn
    git clone https://github.com/jwyang/faster-rcnn.pytorch.git
    cd faster-rcnn.pytorch
    git checkout pytorch-1.0
    cd ..
    mv faster-rcnn.pytorch/* .
    rm -rf faster-rcnn.pytorch
    mv cfgs/res101_ls.yml .
    wget https://www.dropbox.com/s/be0isevd22eikqb/faster_rcnn_1_10_14657.pth
    mv lib/* .
    rm -r lib
    pip3 install -r requirements.txt
    python3 setup.py build develop
    cd ..
fi