import torch
from ultralytics import RTDETR, YOLO


if __name__ == "__main__":
    # RT-DETR
    # torch.use_deterministic_algorithms(False)
    # model = RTDETR(model="./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")

    # YOLO
    model = YOLO(model="/home/image1325_user/ssd_disk1/yuhao_22/YOLOX-SwinTransformer/ultralytics/ultralytics/cfg/models/mymodels/yolov8-afpnMSSCA-yolov8.yaml")
    # YOLO("model.pt")  use pre-trained model if available'

    model.info()  # display model information
    model.train(data="/home/image1325_user/ssd_disk1/yuhao_22/YOLOX-SwinTransformer/data3.yaml",
               epochs=400, device='3', batch=64, name="yam1sirst", single_cls=True)  # train the model