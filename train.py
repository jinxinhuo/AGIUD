import torch
from ultralytics import RTDETR, YOLO


if __name__ == "__main__":
    # RT-DETR
    # torch.use_deterministic_algorithms(False)
    # model = RTDETR(model="./ultralytics/cfg/models/mymodels/fmvit-afpn-rtdetr.yaml")

    # YOLO
    model = RTDETR(model="ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml")
    # YOLO("model.pt")  use pre-trained model if available'
    model.info()  # display model information
    model.train(data="./ultralytics/cfg/datasets/Anti-UAV.yaml",
               epochs=50, device='0', batch=16, name="rtdetr-resnet50", weight_decay=0.001, single_cls=True)  # train the model