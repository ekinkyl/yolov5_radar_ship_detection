import open3d as o3d
import numpy as np
import cv2

IMG_NUM = '000851'

# Load the PCD file
pcd = o3d.io.read_point_cloud(
    f"/raid/yolov5/{IMG_NUM}.pcd"
)

# Get XYZ points
points = np.asarray(pcd.points)

# Round to integers and keep only x,y
xy_points = [(round(p[0]), round(p[1])) for p in points]

# Compute bounding box
x_min = min(p[0] for p in xy_points)
x_max = max(p[0] for p in xy_points)
y_min = min(p[1] for p in xy_points)
y_max = max(p[1] for p in xy_points)

o3d.visualization.draw_geometries([pcd])


# Load image
img = cv2.imread(f"/raid/yolov5/radar_images/000850.png
")

if img is None:
    raise FileNotFoundError("Image not found!")

cx, cy, _ = img.shape
print(img.shape)
x_min = cx // 2 - x_min
x_max = cx // 2 - x_max
y_min += cy // 2
y_max += cy // 2

x_min,x_max,y_min,y_max = y_min,y_max,x_min,x_max

print(f"Bounding box: ({x_min},{y_min}) → ({x_max},{y_max})")

# Draw bounding box (blue rectangle, thickness=2)
cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

img = cv2.resize(img, (1000,1000), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

# Save or display the result
cv2.imwrite(f"{IMG_NUM}_bbox.png", img)   # save new image
cv2.imshow("Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
