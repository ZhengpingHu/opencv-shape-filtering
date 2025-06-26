from ultralytics import YOLO
import cv2

model = YOLO("lander.pt", task="pose")
img = cv2.imread("./00029.jpg")[..., ::-1]  # 一张 val 图片
results = model(img, conf=0.05)[0]

print("Boxes:", results.boxes)
print("Keypoints:", results.keypoints)

rendered = results.plot()
cv2.imshow("Test", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
