from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolo11n-obb.pt")
    model.train(
        data="D:/Git/opencv-shape-filtering/obb/data.yaml",  # 绝对路径
        imgsz=640,
        epochs=100,
        batch=8,
        task="obb"
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 必需
    main()
