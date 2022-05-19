pip install -r PaddleDetection/requirements.txt
unalias cp
cp model/v2net.py PaddleDetection/ppdet/modeling/backbones/
cp model/__init__.py PaddleDetection/ppdet/modeling/backbones/
cp ../../../configs/layout_analysis/datasets/publaynet_detection.yml PaddleDetection/configs/datasets/
cp ../../../configs/layout_analysis/cascade_rcnn/cascade_rcnn_v2.yml PaddleDetection/configs/datasets/cascade_rcnn/
cp ../../../configs/layout_analysis/cascade_rcnn/_base_ PaddleDetection/configs/datasets/cascade_rcnn/
cd PaddleDetection

