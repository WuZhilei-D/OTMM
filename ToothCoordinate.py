import open3d as o3d
import numpy as np
import os


def align_model_to_target_direction(mesh, current_direction, target_direction):
    current_dir = np.array(current_direction) / np.linalg.norm(current_direction)
    target_dir = np.array(target_direction) / np.linalg.norm(target_direction)
    cos_theta = np.dot(current_dir, target_dir)
    if np.isclose(cos_theta, 1.0):
        print("Already aligned with the target direction.")
        return mesh
    elif np.isclose(cos_theta, -1.0):
        rotation_axis = np.array([1, 0, 0]) if not np.isclose(current_dir[0], 1.0) else np.array([0, 1, 0])
        rotation_matrix = np.eye(3) - 2 * np.outer(rotation_axis, rotation_axis)
        mesh.rotate(rotation_matrix, center=(0, 0, 0))
        return mesh

    rotation_axis = np.cross(current_dir, target_dir)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = -np.arccos(cos_theta)

    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    mesh.rotate(rotation_matrix, center=(0, 0, 0))
    return mesh


tooth_files = ['11.stl', '12.stl', '13.stl', '14.stl',
               '15.stl', '16.stl', '17.stl', '21.stl',
               '22.stl', '23.stl', '24.stl', '25.stl',
               '26.stl', '27.stl',
               '31.stl', '32.stl', '33.stl', '34.stl',
               '35.stl', '36.stl', '37.stl', '41.stl',
               '42.stl', '43.stl', '44.stl', '45.stl',
               '46.stl', '47.stl']
output_dir = "translated_teeth"
os.makedirs(output_dir, exist_ok=True)
input_dir = "mesh/"

centroids = []
for file in tooth_files:
    stl_file_path = os.path.join(input_dir, file)
    tooth_mesh = o3d.io.read_triangle_mesh(stl_file_path)
    tooth_points = np.asarray(tooth_mesh.vertices)
    centroid = np.mean(tooth_points, axis=0)
    centroids.append(centroid)
average_centroid = np.mean(centroids, axis=0)
ori = average_centroid
CL = centroids[0]
CR = centroids[7]
CLl = centroids[14]
CRl = centroids[21]
A = (CL + CR + CLl + CRl) / 4
Z_axis = (A - ori)
Z_axis /= np.linalg.norm(Z_axis)
current_z = np.array([0.0, 0.0, 1.0])
for file in tooth_files:
    stl_file_path = os.path.join(input_dir, file)
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    aligned_mesh = align_model_to_target_direction(mesh, current_z, Z_axis)
    aligned_mesh.compute_triangle_normals()
    output_file = os.path.join(output_dir, os.path.basename(file))
    o3d.io.write_triangle_mesh(output_file, aligned_mesh)
    print(f"已保存Z轴变换后的牙齿模型: {output_file}")

centroids.clear()
for file in tooth_files:
    stl_file_path = os.path.join(output_dir, file)
    tooth_mesh = o3d.io.read_triangle_mesh(stl_file_path)
    tooth_points = np.asarray(tooth_mesh.vertices)
    centroid = np.mean(tooth_points, axis=0)
    centroids.append(centroid)
average_centroid = np.mean(centroids, axis=0)
ori = average_centroid
CL = centroids[0]
CR = centroids[7]
CLl = centroids[14]
CRl = centroids[21]
A = (CL + CR + CLl + CRl) / 4
Z_axis = (A - ori)
Z_axis /= np.linalg.norm(Z_axis)
B = centroids[11]
C = centroids[4]
D = centroids[18]
E = centroids[25]
FB = (B + D) / 2
FC = (C + E) / 2
CB = FB - FC
Y_axis = np.cross(Z_axis, CB)
Y_axis /= np.linalg.norm(Y_axis)

current_y = np.array([0.0, 1.0, 0.0])

for file in tooth_files:
    stl_file_path = os.path.join(output_dir, file)
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    aligned_mesh = align_model_to_target_direction(mesh, current_y, Y_axis)
    aligned_mesh.compute_triangle_normals()
    output_file = os.path.join(output_dir, os.path.basename(file))
    o3d.io.write_triangle_mesh(output_file, aligned_mesh)
    print(f"已保存Y轴变换后的牙齿模型: {output_file}")

for file in tooth_files:
    stl_file_path = os.path.join(output_dir, file)
    tooth_mesh = o3d.io.read_triangle_mesh(stl_file_path)
    tooth_points = np.asarray(tooth_mesh.vertices)
    centroid = np.mean(tooth_points, axis=0)
    centroids.append(centroid)
target_origin = np.mean(centroids, axis=0)
print(f"所有牙齿的平均质心: {average_centroid}")

translation_matrix = np.eye(4)
translation_matrix[:3, 3] = -target_origin

for file in tooth_files:
    stl_file_path = os.path.join(output_dir, file)
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh.transform(translation_matrix)
    mesh.compute_triangle_normals()
    output_file = os.path.join(output_dir, os.path.basename(file))
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"已保存平移后的牙齿模型: {output_file}")

print(f"所有处理后的牙齿模型已保存到目录: {output_dir}")

