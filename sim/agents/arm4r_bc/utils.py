import torch
from typing import Union, List, Tuple, Literal
import json
import numpy as np
from scipy.spatial.transform import Rotation


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows

    vectors: batchsize, N, D

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]

    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return basis

def euler_to_rot_6d(euler : np.ndarray, format="XYZ") -> np.ndarray:
    """
    Convert euler angles to 6d representation
    euler: N, 3
    """
    rot_mat = Rotation.from_euler(format, euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_mat_to_rot_6d(rot_mat : np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :] # N, 2, 3
    return rot_6d.reshape(-1, 6) # N, 6

def quat_to_rot_6d(quat : np.ndarray, format : str = "xyzw") -> np.ndarray:
    """
    Convert quaternion to 6d representation
    quat: N, 4
    robomimic:
    https://mujoco.readthedocs.io/en/2.2.1/programming.html#:~:text=To%20represent%203D%20orientations%20and,cos(a%2F2).
    To represent 3D orientations and rotations, MuJoCo uses unit quaternions - namely 4D unit vectors arranged as q = (w, x, y, z).
    Here (x, y, z) is the rotation axis unit vector scaled by sin(a/2), where a is the rotation angle in radians, and w = cos(a/2).
    Thus the quaternion corresponding to a null rotation is (1, 0, 0, 0). This is the default setting of all quaternions in MJCF.
    """
    assert format in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    if format == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_quat(rot_6d : np.ndarray, format : str = "xyzw") -> np.ndarray:
    """
    Convert 6d representation to quaternion
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    quat = Rotation.from_matrix(rot_mat).as_quat()
    if format == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat


def rot_6d_to_rot_mat(rot_6d : np.ndarray) -> np.ndarray:
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat