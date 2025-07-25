import numpy as np
import cv2

def get_grasp_point_from_mask(mask: np.ndarray, depth: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])  # 几何中心

    mask_region = (mask > 0)
    depth_region = depth[mask_region]
    median_depth = np.median(depth_region)

    return (cx, cy), median_depth

def deproject_pixel_to_camera(cx, cy, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    X = (cx - cx0) * depth / fx
    Y = (cy - cy0) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def transform_camera_to_world(point_cam, R, t):
    return R @ point_cam + t

