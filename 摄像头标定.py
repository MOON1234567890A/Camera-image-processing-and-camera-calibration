import numpy as np
import json
from scipy.linalg import rq


def get_location(projection_matrix):
    M = projection_matrix[:, :3]
    L = np.sqrt(np.sum(M ** 2, axis=1)) @ np.array([0.5, 0.5, 0])
    Dm = np.diag([1, 1, L])

    Km, R = rq(Dm @ M)
    E = np.linalg.inv(Km) @ (Dm @ projection_matrix)
    Ic = np.linalg.inv(R) @ E
    return (Ic[0, 3], Ic[1, 3], Ic[2, 3])


def get_intrinsic_matrix_and_rot(projection_matrix):
    # 提取前三列形成矩阵 M
    M = projection_matrix[:, :3]

    # 使用 RQ 分解获取 K 和 R
    K, R = rq(M)

    # 将 K 和 R 中的元素转换为保留17位小数转为string
    K_formatted = np.array([[f"{elem:.17f}" for elem in row] for row in K])
    R_formatted = np.array([[f"{elem:.17f}" for elem in row] for row in R])
    return K_formatted, R_formatted


def compute_projection_matrix_qr(points_3d_hom, points_2d_hom):
    num_points = points_3d_hom.shape[0]
    A = []

    for i in range(num_points):
        X, Y, Z, W = points_3d_hom[i]
        x, y, w = points_2d_hom[i]
        A.append([0, 0, 0, 0, -w * X, -w * Y, -w * Z, -w * W, y * X, y * Y, y * Z, y * W])
        A.append([w * X, w * Y, w * Z, w * W, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x * W])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    P /= P[-1, -1]
    return P


# 解析JSON数据
with open('1.json', 'r') as file:
    data = json.load(file)

# 提取3D点和2D点
points_3d = []
points_2d = []
for point in data["data"]:
    # 3D点
    cartesian = point["cartesian"]
    points_3d.append([cartesian["x"], cartesian["y"], cartesian["z"], -1])
    # 2D点
    points_2d.append([point["x"], point["y"], 1])

# 转换为NumPy数组
points_3d = np.array(points_3d)
points_2d = np.array(points_2d)
P = compute_projection_matrix_qr(points_3d, points_2d)
print("投影矩阵 P:\n", P)
K, R = get_intrinsic_matrix_and_rot(P)
print("旋转矩阵 R:\n", R)
G = get_location(P)
print("摄像头三维坐标:\n", G)
