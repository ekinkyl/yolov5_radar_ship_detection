import cv2

img_path = "/raid/yolov5/newDataset/images/test/X_1724241962.png"
label_path = "/raid/yolov5/newDataset/labels/test/X_1724241962.txt"

img = cv2.imread(img_path)
H, W = img.shape[:2]
print("Image size:", W, H)

with open(label_path) as f:
    for line in f:
        cls, x, y, w, h = map(float, line.split())
        print("Raw normalized:", x, y, w, h)
        print("Pixel coords:", x * W, y * H, w * W, h * H)
        break
