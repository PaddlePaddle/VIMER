pip install -r PaddleDetection/requirements.txt
cp model/* PaddleDetection/ppdet/modeling/backbones/
cp ../../../configs/layout_analysis/datasets/publaynet_detection.yml PaddleDetection/configs/datasets/
cp ../../../configs/layout_analysis/cascade_rcnn/cascade_rcnn_v2.yml PaddleDetection/configs/cascade_rcnn/
cp ../../../configs/layout_analysis/cascade_rcnn/_base_/* PaddleDetection/configs/cascade_rcnn/_base_/
