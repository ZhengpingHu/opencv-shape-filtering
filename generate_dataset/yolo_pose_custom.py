from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
model.train(
    data='pose_lander/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.001,
    hyp='custom_pose.yaml',
    project='runs/pose',
    name='lander_pose25'
)
