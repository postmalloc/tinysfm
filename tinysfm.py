import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path = sys.argv[1]  # Data directory path. Should contain camera intrinsics in `K.txt`.
fnames = glob.glob(f'{data_path}/*.jpg')
fnames.sort()  # Order of capture is important for incremental SFM!

K = np.loadtxt(f'{data_path}/K.txt', dtype=np.float)  # Load intrinsics
kp_prev, desc_prev = None, None
Rt_prev = np.append(np.eye(3), np.zeros((3, 1)), axis=-1)  # Init extrinsics for current frame
Rt = np.zeros((3, 4))  # Init extrinsics for next frame
P_prev = K @ Rt_prev  # Init projection matrix for current frame
P = np.zeros((3, 4))  # Init projection matrix for next frame
pt_cld = np.empty((3, 1))

for i in range(len(fnames)):
    img = cv2.imread(fnames[i])
    det = cv2.xfeatures2d.SIFT_create()  # Init SIFT detector
    kp, desc = det.detectAndCompute(img, None)  # Extract keypoints & their descriptors
    if i == 0:  # If first frame, update and skip
        kp_prev, desc_prev = kp, desc
        continue

    matcher = cv2.BFMatcher_create()  # Use a simple bruteforce matcher
    matches = matcher.knnMatch(desc_prev, desc, k=2)  # Get top 2 matches for Lowe's test
    pts_prev, pts = [], []
    for m in matches:
        a, b = m[0], m[1]
        if a.distance / b.distance < 0.80:  # Lowe's ratio test
            pts_prev.append(kp_prev[a.queryIdx].pt)
            pts.append(kp[a.trainIdx].pt)
    pts_prev, pts = np.array(pts_prev), np.array(pts)
    F, mask = cv2.findFundamentalMat(pts_prev, pts, cv2.RANSAC)  # Fundamental Matrix for the two frames
    mask = mask.ravel() == 1
    pts_prev, pts = pts_prev[mask], pts[mask]  # Exploit Epipolar constraint and keep only useful points

    E = K.T @ F @ K  # Find Essential Matrix
    _, R, t, _ = cv2.recoverPose(E, pts_prev, pts, K)  # Get current camera rotation + translation
    Rt[:, :3] = R @ Rt_prev[:, :3]  # Update rotational params
    Rt[:, 3] = Rt_prev[:, 3] + Rt_prev[:, :3] @ t.ravel()  # Update translation params
    P = K @ Rt  # Derive projection matrix for triangulation
    pts_3d = cv2.triangulatePoints(P_prev, P, pts_prev.T, pts.T)  # Find 3D coords from 2D points
    pts_3d = cv2.convertPointsFromHomogeneous(pts_3d.T)[:, 0, :].T  # Homogenous (4D) -> Cartesian (3D)
    pt_cld = np.concatenate([pt_cld, pts_3d], axis=-1)  # Add 3D points to point cloud
    P_prev, Rt_prev, kp_prev, desc_prev = np.copy(P), np.copy(Rt), kp, desc  # Updates for next iteration

fig = plt.figure(1, (10, 10))  # Plot the point cloud
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pt_cld[0], pt_cld[1], pt_cld[2], s=1)
plt.show()
