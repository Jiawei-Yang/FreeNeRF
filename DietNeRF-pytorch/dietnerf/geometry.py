import numpy as np
from scipy.spatial.transform import Rotation
import torch


def slerp(p0, p1, t):
    # https://stackoverflow.com/questions/2879441/how-to-interpolate-rotations
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def interp(pose1, pose2, s):
    """Interpolate between poses as camera-to-world transformation matrices"""
    assert pose1.shape == (3, 4)
    assert pose2.shape == (3, 4)

    # Camera translation 
    C = (1 - s) * pose1[:, -1] + s * pose2[:, -1]
    assert C.shape == (3,)

    # Rotation from camera frame to world frame
    R1 = Rotation.from_matrix(pose1[:, :3])
    R2 = Rotation.from_matrix(pose2[:, :3])
    R = slerp(R1.as_quat(), R2.as_quat(), s)
    R = Rotation.from_quat(R)
    R = R.as_matrix()
    assert R.shape == (3, 3)
    transform = np.concatenate([R, C[:, None]], axis=-1)
    return torch.tensor(transform, dtype=pose1.dtype)


def interp3(pose1, pose2, pose3, s12, s3):
    return interp(interp(pose1, pose2, s12).cpu(), pose3, s3)
