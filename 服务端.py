# app.py
from flask import Flask, request, jsonify
import json
import numpy as np
from scipy.linalg import rq

app = Flask(__name__)


def get_location(projection_matrix):
    M = projection_matrix[:, :3]
    L = np.sqrt(np.sum(M ** 2, axis=1)) @ np.array([0.5, 0.5, 0])
    Dm = np.diag([1, 1, L])

    Km, R = rq(Dm @ M)
    E = np.linalg.inv(Km) @ (Dm @ projection_matrix)
    Ic = np.linalg.inv(R) @ E
    return (Ic[0, 3], Ic[1, 3], Ic[2, 3])


def get_intrinsic_matrix_and_rot(projection_matrix):
    M = projection_matrix[:, :3]
    K, R = rq(M)
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


@app.route('/process', methods=['POST'])
def process_json():
    if not request.is_json:
        return jsonify({"error": "Request data must be JSON"}), 400

    data = request.json

    if "data" not in data or not isinstance(data["data"], list):
        return jsonify({"error": "Invalid JSON structure, 'data' field is required and must be a list"}), 400

    if len(data["data"]) < 6:
        warning = "Data points fewer than 6 may cause significant errors"

    points_3d = []
    points_2d = []

    for point in data["data"]:
        if not all(k in point for k in ["cartesian", "x", "y"]):
            return jsonify({"error": "Each data point must contain 'cartesian', 'x', and 'y' fields"}), 400

        cartesian = point["cartesian"]
        if not all(k in cartesian for k in ["x", "y", "z"]):
            return jsonify({"error": "Cartesian coordinates must contain 'x', 'y', and 'z' fields"}), 400

        points_3d.append([cartesian["x"], cartesian["y"], cartesian["z"], -1])
        points_2d.append([point["x"], point["y"], 1])

    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    P = compute_projection_matrix_qr(points_3d, points_2d)
    K, R = get_intrinsic_matrix_and_rot(P)
    G = get_location(P)

    response_data = {
        "projection_matrix": P.tolist(),
        "rotation_matrix": R.tolist(),
        "camera_coordinates": G
    }

    if len(data["data"]) < 6:
        response_data["warning"] = warning

    with open('output.json', 'w') as file:
        json.dump(response_data, file, indent=4)

    return jsonify({"message": "Data processed and saved to output.json", "data": response_data})


if __name__ == '__main__':
    app.run(debug=True)
