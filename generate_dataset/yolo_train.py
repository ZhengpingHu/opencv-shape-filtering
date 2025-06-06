import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from ultralytics import YOLO
import multiprocessing


def main():
    model = YOLO("yolo11n-obb.pt")
    model.train(
    data="D:/Git/opencv-shape-filtering/generate_dataset/dataset/data.yaml", 
    imgsz=640,
    epochs=100,
    batch=8,
    workers=0,
    task="obb",
    verbose=True,
    name="train_fixed"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 必需
    main()
