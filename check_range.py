import glob, numpy as np, open3d as o3d

ship_pcd_root = "ship"
all_x, all_y = [], []

for pcd_file in glob.glob(ship_pcd_root + "/id_*/" + "*.pcd"):
    pcd = o3d.io.read_point_cloud(pcd_file)
    pts = np.asarray(pcd.points)
    if pts.shape[0] > 0:
        all_x.extend(pts[:,0])
        all_y.extend(pts[:,1])

print("Global X range:", min(all_x), max(all_x))
print("Global Y range:", min(all_y), max(all_y))
