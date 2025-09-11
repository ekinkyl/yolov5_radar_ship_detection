import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import open3d as o3d

frame = "000850"
pcd_file = f"ship/id_04/{frame}.pcd"
img_file = f"radar_images/{frame}.png"

pcd = o3d.io.read_point_cloud(pcd_file)
pts = np.asarray(pcd.points)
img = np.array(Image.open(img_file))
H, W = img.shape[:2]

X_MIN, X_MAX = 40, 170
Y_MIN, Y_MAX = -35, 100

# project into image pixels
u = (pts[:,0] - X_MIN) / (X_MAX - X_MIN) * W
v = (pts[:,1] - Y_MIN) / (Y_MAX - Y_MIN) * H

plt.imshow(img, cmap="gray")
plt.scatter(u, v, s=5, c='r')
plt.title(f"Radar points projected on {frame}.png")
plt.show()
