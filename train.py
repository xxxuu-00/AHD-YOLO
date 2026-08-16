import warnings, os, sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    yaml_path = ''


    model = YOLO(yaml_path)

    model.train(data='',
                cache=False,
                imgsz=416,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=1,
                device=os.environ.get("CUDA_VISIBLE_DEVICES", 0),
                optimizer='MuSGD' if 'yolo26' in yaml_path else 'SGD',
                patience=0,
                amp=False,

                cos_lr=False,
                save_period=-1,
                project='',
                name='',
                cls_loss='bce',
                iou_loss='ciou',
                iou_aux='none',
                iou_aux_ratio=0.5,
                )
