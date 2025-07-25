import numpy as np
import matplotlib.pyplot as plt

# θi is variable; d, a, α are constant
DH_PARAMS = [
    {'d': 0,   'a': 0,   'alpha': np.pi/2},
    {'d': 0,   'a': 431, 'alpha': 0},
    {'d': 149, 'a': 20,  'alpha': np.pi/2},
    {'d': 433, 'a': 0,   'alpha': -np.pi/2},
    {'d': 0,   'a': 0,   'alpha': np.pi/2},
    {'d': 56,  'a': 0,   'alpha': 0}
]

# Joint limits (radians) for PUMA 650 (approximate)
JOINT_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi/2, np.pi/2),
    (-np.pi/2, np.pi/2),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi)
]

def dh_transform(theta, d, a, alpha):
    """Compute individual DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,      ca,     d],
        [0,        0,       0,     1]
    ])

def puma_inverse_kinematics(position, R06):
    """Inverse kinematics solution for PUMA 650 without using external libraries."""
    x, y, z = position
    d1, a2, d3, d4, d6 = 0, 431, 149, 433, 56
    a3 = 20

    # Step 1: Compute wrist center
    P06 = np.array([x, y, z])
    P05 = P06 - d6 * R06[:, 2]
    xw, yw, zw = P05

    # θ1
    theta1 = np.arctan2(yw, xw)

    # r, s for planar triangle
    r = np.sqrt(xw**2 + yw**2)
    s = zw - d1
    D = ((r**2 + s**2 - a2**2 - d3**2) / (2 * a2 * d3))
    D = np.clip(D, -1, 1)  # clip for numerical stability

    theta3 = np.arctan2(np.sqrt(1 - D**2), D)  # elbow down solution

    phi1 = np.arctan2(s, r)
    phi2 = np.arctan2(d3 * np.sin(theta3), a2 + d3 * np.cos(theta3))
    theta2 = phi1 - phi2

    # T03
    A1 = dh_transform(theta1, *[DH_PARAMS[0][k] for k in ['d', 'a', 'alpha']])
    A2 = dh_transform(theta2, *[DH_PARAMS[1][k] for k in ['d', 'a', 'alpha']])
    A3 = dh_transform(theta3, *[DH_PARAMS[2][k] for k in ['d', 'a', 'alpha']])
    T03 = A1 @ A2 @ A3
    R03 = T03[:3, :3]

    R36 = R03.T @ R06
    theta5 = np.arccos(np.clip(R36[2, 2], -1, 1))
    theta4 = np.arctan2(R36[1, 2], R36[0, 2])
    theta6 = np.arctan2(R36[2, 1], -R36[2, 0])

    # Joint array
    joint_angles = np.array([theta1, theta2, theta3, theta4, theta5, theta6])

    # Enforce joint limits
    for i in range(6):
        low, high = JOINT_LIMITS[i]
        if not (low <= joint_angles[i] <= high):
            print(f"Joint {i+1} angle {np.degrees(joint_angles[i]):.2f}° out of range!")
            return None

    return joint_angles

# For visualization
def visualize_joint_angles(joint_angles):
    joint_labels = [f"θ{i+1}" for i in range(6)]
    degrees = np.degrees(joint_angles)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(joint_labels, degrees, color='skyblue')
    plt.title("PUMA 650 Inverse Kinematics Solution (Degrees)")
    plt.ylabel("Angle (degrees)")
    plt.ylim(-180, 180)
    for bar, angle in zip(bars, degrees):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{angle:.1f}°", ha='center', va='bottom')
    plt.grid(True)
    plt.show()

