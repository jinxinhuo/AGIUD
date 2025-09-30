import torch
from ultralytics import YOLO
# from ray import tune
import sys
import os

# 添加包含自定义模块的目录到 Python 路径
custom_module_path = "/home/image1325_user/ssd_disk1/yuhao_22/YOLOX-SwinTransformer"
sys.path.insert(0, custom_module_path)



if __name__ == "__main__":
    # RT-DETR
    # torch.use_deterministic_algorithms(False)
    # model = RTDETR(model="./ultralytics/cfg/models/mymodels/fmvit-afpn-rtdetr.yaml")

    # YOLO
    model = YOLO(model="/home/image1325_user/ssd_disk1/yuhao_22/YOLOX-SwinTransformer/ultralytics/runs/detect/yam1sirst2/weights/best.pt")
    # YOLO("model.pt")  use pre-trained model if available'

    model.info()  # display model information
    # metrics = model.val(data="/home/image1325_user/ssd_disk1/yuhao_22/YOLOX-SwinTransformer/ultralytics/ultralytics/cfg/datasets/Anti-UAV.yaml", device='3')
    metrics = model.val(data="/home/image1325_user/ssd_disk1/yuhao_22/YOLOX-SwinTransformer/data3.yaml", device='3')# train the model
    print(metrics.box.maps)
    print(f"AP75: {metrics.box.map75}")
    print(f"AP50: {metrics.box.map50}")
    print(f"AP50:95 (mAP): {metrics.box.map}")
    print(f"Precision (P): {metrics.box.mp}")
    print(f"Recall (R): {metrics.box.mr}")