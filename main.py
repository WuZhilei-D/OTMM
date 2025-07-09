import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from CameraPoseOptimizer import CameraPoseOptimizer
from TeethPoseOptimizer import TeethPoseOptimizer
from scipy.spatial.transform import Rotation as RR
from lmfit import Parameters
from Analyze import ToothModelEvaluator

target_width = 1450


def subdivide_mesh(mesh, target_vertex_count):
    while len(np.asarray(mesh.vertices)) < target_vertex_count:
        mesh = mesh.subdivide_loop(number_of_iterations=1)
    return mesh


def load_mesh(model_folder, tooth_ids, target_vertex_count=12000):
    tooth_vertices = {}
    for tooth_id in tooth_ids:
        obj_file_path = os.path.join(model_folder, f"{tooth_id}.stl")
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        mesh.remove_duplicated_vertices()
        if not mesh.is_empty():
            if len(np.asarray(mesh.vertices)) < target_vertex_count:
                mesh = subdivide_mesh(mesh, target_vertex_count)
            mesh.remove_duplicated_vertices()
            tooth_vertices[tooth_id] = np.asarray(mesh.vertices)
        else:
            print(f"警告: 牙齿模型 {tooth_id} 加载失败或为空。")
    return tooth_vertices


def project_points(view_name, points_3d, camera_matrix, rvec, tvec, show=False):
    avg_depth = {}
    points2dimension = []
    for tooth_id, vertices in points_3d.items():
        R = RR.from_rotvec(rvec).as_matrix().T
        extrViewMat = np.vstack([R, tvec])
        X_homo = np.concatenate([vertices, np.ones((*vertices.shape[:-1], 1))], axis=-1)
        projected_points = np.matmul(X_homo, extrViewMat)
        avg_depth[tooth_id] = np.mean(projected_points[:, 2])
        points_camera = projected_points / projected_points[..., [2]]
        points_2d_homo = np.matmul(points_camera, camera_matrix)
        points_2d = points_2d_homo[..., :2]
        points2dimension.append(points_2d)
    sorted_tooth_ids = sorted(avg_depth.keys(), key=lambda x: avg_depth[x])
    p_dimension = np.vstack(points2dimension)
    min_x, min_y = np.min(p_dimension, axis=0)
    max_x, max_y = np.max(p_dimension, axis=0)
    margin = 5
    img_width = int(max_x - min_x + 2 * margin)
    img_height = int(max_y - min_y + 2 * margin)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    visible_points_2d = {}

    for tooth_id in sorted_tooth_ids:
        vertices = points_3d[tooth_id]
        R = RR.from_rotvec(rvec).as_matrix().T
        extrViewMat = np.vstack([R, tvec])
        X_homo = np.concatenate([vertices, np.ones((*vertices.shape[:-1], 1))], axis=-1)
        projected_points = np.matmul(X_homo, extrViewMat)
        invalid_z_val = projected_points[..., 2] < 0
        projected_points[..., 2][invalid_z_val] = 0.0
        points_camera = projected_points / projected_points[..., [2]]
        points_2d_homo = np.matmul(points_camera, camera_matrix)
        points_2d = points_2d_homo[..., :2]

        visible_indices = []
        for i, pt in enumerate(points_2d):
            u = int(pt[0] - min_x + margin)
            v = int(pt[1] - min_y + margin)
            if mask[v, u] != 255:
                mask[v, u] = 255
                visible_indices.append(i)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        visible_points_2d[tooth_id] = points_2d[visible_indices]
    merged_contours, merged_contour_normals = extract_contours(visible_points_2d, view_name)
    sampled_contours = []
    sampled_normals = []
    for contour, normal in zip(merged_contours, merged_contour_normals):
        sampled_points, sampled_normals_group = uniform_downsample(np.array(contour), np.array(normal))
        sampled_contours.append(sampled_points)
        sampled_normals.append(sampled_normals_group)
    contour_points = np.vstack(sampled_contours)
    contour_normals = np.vstack(sampled_normals)
    if show:
        plt.cla()
        plt.plot(contour_points[:, 0], contour_points[:, 1], 'r.', label='contour and normal', markersize=1)
        for point, normal in zip(contour_points, contour_normals):
            start_x, start_y = point
            end_x = start_x + normal[0] * 100
            end_y = start_y + normal[1] * 100
            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                      head_width=1, head_length=2, fc='blue', ec='blue', alpha=0.6)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
    return sampled_contours, sampled_normals


def load_image_contour(image_folder, view_name, show=False):
    image_path = os.path.join(image_folder, f"{view_name}.png")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_height, original_width = image.shape[:2]
    scale_factor = target_width / original_width
    target_height = int(original_height * scale_factor)
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    height, width = image.shape[:2]
    merged_contours = []
    merged_contour_normals = []
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]
    for color in unique_colors:
        mask = cv2.inRange(image, color, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_points = contour[:, 0, :]
            contour_normals = init_edge_mask_normals(contour_points, False)
            merged_contours.append(contour_points)
            merged_contour_normals.append(contour_normals)
    sampled_contours = []
    sampled_normals = []
    for contour, normal in zip(merged_contours, merged_contour_normals):
        sampled_points, sampled_normals_group = uniform_downsample(np.array(contour), np.array(normal))
        sampled_contours.append(sampled_points)
        sampled_normals.append(sampled_normals_group)
    contour_points = np.vstack(sampled_contours)
    contour_normals = np.vstack(sampled_normals)
    if show:
        plt.cla()
        plt.plot(contour_points[:, 0], contour_points[:, 1], 'r.', label='contour and normal', markersize=1)
        for point, normal in zip(contour_points, contour_normals):
            start_x, start_y = point
            end_x = start_x + normal[0] * 100
            end_y = start_y + normal[1] * 100
            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                      head_width=1, head_length=2, fc='blue', ec='blue', alpha=0.6)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    image_path = os.path.join(image_folder, "original", f"{view_name}.jpg")
    oral_image = cv2.imread(image_path)
    oral_image = cv2.resize(oral_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    for x, y in contour_points:
        cv2.circle(oral_image, (x, y), 9, (55, 108, 73), -1)  # 深绿色
    output_file_path2 = os.path.join(image_folder, "original", f"{view_name}_overlay.png")
    cv2.imwrite(output_file_path2, oral_image)

    return sampled_contours, sampled_normals, height, width


def uniform_downsample(points, normals, num_samples=30):
    if len(points) <= num_samples:
        return points, normals
    indices = np.linspace(0, len(points) - 1, num_samples, dtype=np.int32)  # 均匀采样索引
    sampled_points = points[indices]
    sampled_normals = normals[indices]
    return sampled_points, sampled_normals


def extract_contours(visible_points_2d, view_name):
    merged_contours = []
    merged_contour_normals = []
    for tooth_id, points_2d in visible_points_2d.items():
        if points_2d is None or len(points_2d) == 0:
            print(f"Warning: No points available for tooth {tooth_id} in view {view_name}. Skipping...")
            continue
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)
        margin = 5
        img_width = int(max_x - min_x + 2 * margin)
        img_height = int(max_y - min_y + 2 * margin)
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for i, pt in enumerate(points_2d):
            x_img = int(pt[0] - min_x + margin)
            y_img = int(pt[1] - min_y + margin)
            mask[y_img, x_img] = 255
        kernel = np.ones((5, 5), np.uint8)
        closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)  # 选择面积最大的轮廓
            contour_points = contour[:, 0, :]  # 提取轮廓的点集
            contour_points[:, 0] = contour_points[:, 0] + min_x - margin
            contour_points[:, 1] = contour_points[:, 1] + min_y - margin
            contour_normals = init_edge_mask_normals(contour_points, False)
            merged_contours.append(contour_points)
            merged_contour_normals.append(contour_normals)
    return merged_contours, merged_contour_normals


def init_edge_mask_normals(vertices_xy, show=False):
    """估算轮廓法向量并调整方向."""
    M = len(vertices_xy)
    pcd = o3d.geometry.PointCloud()
    jitter = 1e-3 * np.random.randn(M, 2)
    vertices_xy_jittered = vertices_xy + jitter
    pcd.points = o3d.utility.Vector3dVector(
        np.hstack([vertices_xy_jittered, 20 * np.random.rand(M, 1)])
    )
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=20)
    normals_xy = np.asarray(pcd.normals)[:, :2]
    pcd.normals = o3d.utility.Vector3dVector(
        np.hstack([normals_xy, np.zeros((M, 1))])
    )
    pcd.normalize_normals()
    centroid = np.mean(vertices_xy, axis=0)
    adjusted_normals = []
    for i, point in enumerate(vertices_xy):
        normal = np.array(pcd.normals[i][:2])
        to_center = centroid - point
        if np.dot(normal, to_center) > 0:
            normal = -normal
        adjusted_normals.append(normal)
    if show:
        adjusted_normals_3d = np.hstack([adjusted_normals, np.zeros((M, 1))])
        pcd.normals = o3d.utility.Vector3dVector(adjusted_normals_3d)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="image edge normals estimation",
            width=800,
            height=600,
            left=50,
            top=50,
            point_show_normal=True,
        )
    return adjusted_normals


def overlay_contours_camera(camera_params, iter):  # 测试用
    """优化单视图的相机姿态."""
    for view_name in camera_params.keys():
        tooth_ids = tooth_views.get(view_name, [])
        points_3d = {}
        for tooth_id in tooth_ids:
            points_3d[tooth_id] = tooth_points_3d[tooth_id]
        real_contour = contours_and_normals[view_name]['contour']
        params = camera_params[view_name]
        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']
        rvec = np.array([params['rvec_0'], params['rvec_1'], params['rvec_2']])
        tvec = np.array([params['tvec_0'], params['tvec_1'], params['tvec_2']])
        camera_matrix = np.array([[fx, 0, 0], [0, fy, 0], [cx, cy, 1]])
        projected_contour, projected_contour_normal = project_points(view_name, points_3d, camera_matrix, rvec,
                                                                     tvec, False)
        real_contour = np.vstack(real_contour)
        projected_contour = np.vstack(projected_contour)
        image_width, image_height = image_dimensions[view_name]
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8) * 255
        for x, y in real_contour:
            if 0 <= x < image_width and 0 <= y < image_height:
                cv2.circle(image, (x, y), 9, (55, 108, 73), -1)
        for x, y in projected_contour:
            if 0 <= x < image_width and 0 <= y < image_height:
                cv2.circle(image, (x, y), 9, (180, 138, 103), -1)
        log_dir = "log/CameraPoseOptimizer"
        output_file_path = os.path.join(log_dir, f"{iter}-{view_name}_optimized.png")
        cv2.imwrite(output_file_path, image)


def overlay_contours_teeth(camera_params, teeth_pose, iter):
    for view_name in tooth_views.keys():
        tooth_ids = tooth_views.get(view_name, [])
        points_3d = {}
        for tooth_id in tooth_ids:
            params = teeth_pose[tooth_id]
            tooth_rvec = np.array([params['rvec_0'], params['rvec_1'], params['rvec_2']])
            tooth_tvec = np.array([params['tvec_0'], params['tvec_1'], params['tvec_2']])
            current_tooth_vertices = tooth_points_3d[tooth_id]
            Rvec = RR.from_rotvec(tooth_rvec).as_matrix().T
            PoseMat = np.vstack([Rvec, tooth_tvec])
            tooth_homo = np.concatenate([current_tooth_vertices, np.ones((*current_tooth_vertices.shape[:-1], 1))],
                                        axis=-1)
            points_teeth = np.matmul(tooth_homo, PoseMat)
            points_3d[tooth_id] = points_teeth
        real_contour = contours_and_normals[view_name]['contour']
        camera_param = camera_params[view_name]
        fx, fy, cx, cy = camera_param['fx'], camera_param['fy'], camera_param['cx'], camera_param['cy']
        rvec = np.array([camera_param['rvec_0'], camera_param['rvec_1'], camera_param['rvec_2']])
        tvec = np.array([camera_param['tvec_0'], camera_param['tvec_1'], camera_param['tvec_2']])
        camera_matrix = np.array([[fx, 0, 0], [0, fy, 0], [cx, cy, 1]])
        projected_contour, projected_contour_normal = project_points(view_name, points_3d, camera_matrix,
                                                                     rvec,
                                                                     tvec, False)
        real_contour = np.vstack(real_contour)
        projected_contour = np.vstack(projected_contour)
        image_width, image_height = image_dimensions[view_name]
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8) * 255

        image_path = os.path.join(image_folder, "original", f"{view_name}.jpg")
        oral_image = cv2.imread(image_path)
        original_height, original_width = oral_image.shape[:2]
        scale_factor = target_width / original_width
        target_height = int(original_height * scale_factor)
        oral_image = cv2.resize(oral_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        for x, y in real_contour:
            cv2.circle(image, (x, y), 9, (55, 108, 73), -1)
        for x, y in projected_contour:
            cv2.circle(image, (x, y), 9, (180, 138, 103), -1)
            cv2.circle(oral_image, (x, y), 9, (180, 138, 103), -1)
        log_dir = "log/TeethPoseOptimizer"
        output_file_path = os.path.join(log_dir, f"{iter}-{view_name}_teeth_pose_optimized.png")
        cv2.imwrite(output_file_path, image)
        output_file_path2 = os.path.join(log_dir, f"{iter}-{view_name}_overlay.png")
        cv2.imwrite(output_file_path2, oral_image)


def initialize_params(image_dimensions, view_name):
    params = Parameters()
    width, height = image_dimensions[view_name]
    cx_default = width / 2
    cy_default = height / 2
    if view_name == "bottom":
        params.add('fx', value=3500, min=0, max=100000)
        params.add('fy', value=3500, min=0, max=100000)
        params.add('cx', value=cx_default, min=0, max=100000)
        params.add('cy', value=cy_default, min=0, max=100000)
        params.add('rvec_0', value=-0.6 * np.pi, min=-1000, max=1000)
        params.add('rvec_1', value=0, min=-1000, max=1000)
        params.add('rvec_2', value=0, min=-1000, max=1000)
        params.add('tvec_0', value=0, min=-1000, max=1000)
        params.add('tvec_1', value=0, min=-1000, max=1000)
        params.add('tvec_2', value=180, min=-1000, max=1000)
    elif view_name == "top":
        params.add('fx', value=3500, min=0, max=100000)
        params.add('fy', value=3500, min=0, max=100000)
        params.add('cx', value=cx_default, min=0, max=100000)
        params.add('cy', value=cy_default, min=0, max=100000)
        params.add('rvec_0', value=0.6 * np.pi, min=-1000, max=1000)
        params.add('rvec_1', value=0, min=-1000, max=1000)
        params.add('rvec_2', value=0, min=-1000, max=1000)
        params.add('tvec_0', value=0, min=-1000, max=1000)
        params.add('tvec_1', value=0, min=-1000, max=1000)
        params.add('tvec_2', value=180, min=-1000, max=1000)
    elif view_name == "front":
        params.add('fx', value=5000, min=0, max=100000)
        params.add('fy', value=5000, min=0, max=100000)
        params.add('cx', value=cx_default, min=0, max=100000)
        params.add('cy', value=cy_default, min=0, max=100000)
        params.add('rvec_0', value=np.pi, min=-1000, max=1000)
        params.add('rvec_1', value=0, min=-1000, max=1000)
        params.add('rvec_2', value=0, min=-1000, max=1000)
        params.add('tvec_0', value=0, min=-1000, max=1000)
        params.add('tvec_1', value=0, min=-1000, max=1000)
        params.add('tvec_2', value=260, min=-1000, max=1000)
    elif view_name == "left":
        params.add('fx', value=2500, min=0, max=100000)
        params.add('fy', value=2500, min=0, max=100000)
        params.add('cx', value=cx_default, min=0, max=100000)
        params.add('cy', value=cy_default, min=0, max=100000)
        params.add('rvec_0', value=0.9 * np.pi, min=-1000, max=1000)
        params.add('rvec_1', value=0, min=-1000, max=1000)
        params.add('rvec_2', value=-0.4 * np.pi, min=-1000, max=1000)
        params.add('tvec_0', value=-6.5, min=-1000, max=1000)
        params.add('tvec_1', value=0, min=-1000, max=1000)
        params.add('tvec_2', value=120, min=-1000, max=1000)
    elif view_name == "right":
        params.add('fx', value=2500, min=0, max=100000)
        params.add('fy', value=2500, min=0, max=100000)
        params.add('cx', value=cx_default, min=0, max=100000)
        params.add('cy', value=cy_default, min=0, max=100000)
        params.add('rvec_0', value=0.9 * np.pi, min=-1000, max=1000)
        params.add('rvec_1', value=0, min=-1000, max=1000)
        params.add('rvec_2', value=0.4 * np.pi, min=-1000, max=1000)
        params.add('tvec_0', value=6.5, min=-1000, max=1000)
        params.add('tvec_1', value=0, min=-1000, max=1000)
        params.add('tvec_2', value=120, min=-1000, max=1000)
    return params


def initialize_teeth_pose(tooth_id):
    params = Parameters()

    params.add('rvec_1', value=0, min=-500, max=500)
    params.add('tvec_0', value=0, min=-1000, max=1000)
    # 如果牙齿 ID 为 17、27、37、47，固定 tvec_1 为 0
    if tooth_id in {17, 27, 37, 47}:
        params.add('tvec_1', value=0, vary=False)  # 固定 Y轴平移，不参与优化
        params.add('rvec_0', value=0, vary=False)  # 不能绕X轴旋转
        params.add('rvec_2', value=0, vary=False)  # 不能绕Z轴旋转
    else:
        params.add('tvec_1', value=0, min=-1000, max=1000)  # 正常优化
        params.add('rvec_0', value=0, min=-500, max=500)
        params.add('rvec_2', value=0, min=-500, max=500)
    params.add('tvec_2', value=0, min=-1000, max=1000)
    return params


if __name__ == "__main__":
    tooth_views = {
        "front": [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26,
                  31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46],
        "left": [11, 21, 22, 23, 24, 25, 26,
                 31, 32, 33, 34, 35, 36, 41],
        "right": [21, 11, 12, 13, 14, 15, 16,
                  41, 42, 43, 44, 45, 46, 31],
        "top": [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
        "bottom": [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    }
    tooth_ids = [
        11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    model_folder = "mesh/"
    image_folder = "label/"
    tooth_points_3d = load_mesh(model_folder, tooth_ids)
    contours_and_normals = {}
    image_dimensions = {}
    camera_params = {}
    teeth_pose = {}
    for view_name in tooth_views.keys():
        real_contour, real_normals, height, width = load_image_contour(image_folder, view_name, False)
        contours_and_normals[view_name] = {
            'contour': real_contour,
            'normals': real_normals
        }
        image_dimensions[view_name] = (width, height)
        camera_params[view_name] = initialize_params(image_dimensions, view_name)
    for tooth_id in tooth_ids:
        teeth_pose[tooth_id] = initialize_teeth_pose(tooth_id)
    overlay_contours_camera(camera_params, -1)
    E_loss = []
    with open("log/optimization_loss.txt", "w") as loss_file:
        for iter in range(3):
            camera_optimizer = CameraPoseOptimizer(tooth_points_3d=tooth_points_3d,
                                                   contours_and_normals=contours_and_normals, teeth_pose=teeth_pose,
                                                   tooth_views=tooth_views, image_dimensions=image_dimensions,
                                                   camera_params=camera_params, iteration=iter)
            temp_camera_params = camera_optimizer.optimize_all_views()
            overlay_contours_camera(temp_camera_params, iter)
            print("相机内外参优化完成")
            teeth_optimizer = TeethPoseOptimizer(tooth_points_3d=tooth_points_3d,
                                                 contours_and_normals=contours_and_normals,
                                                 tooth_views=tooth_views, tooth_ids=tooth_ids,
                                                 camera_params=temp_camera_params, teeth_pose=teeth_pose,
                                                 image_dimensions=image_dimensions, iteration=iter)
            temp_teeth_pose, e_loss = teeth_optimizer.optimize_all_teeth()
            loss_file.write(f"Iteration {iter}, E_loss: {e_loss:.4f}\n")
            loss_file.flush()
            if len(E_loss) >= 1 and (e_loss >= E_loss[-1]):
                print(
                    "Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(
                        E_loss[-2], E_loss[-1], e_loss
                    )
                )
                break
            else:
                camera_params = temp_camera_params
                teeth_pose = temp_teeth_pose
                E_loss.append(e_loss)
            overlay_contours_teeth(camera_params, teeth_pose, iter)
            print("牙齿位姿优化完成")

    for tooth_id in tooth_ids:
        obj_file_path = os.path.join(model_folder, f"{tooth_id}.stl")
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        mesh.remove_duplicated_vertices()
        current_tooth_vertices = np.asarray(mesh.vertices)
        params = teeth_pose[tooth_id]
        tooth_rvec = np.array([params['rvec_0'], params['rvec_1'], params['rvec_2']])
        tooth_tvec = np.array([params['tvec_0'], params['tvec_1'], params['tvec_2']])
        Rvec = RR.from_rotvec(tooth_rvec).as_matrix().T
        PoseMat = np.vstack([Rvec, tooth_tvec])
        tooth_homo = np.concatenate([current_tooth_vertices, np.ones((*current_tooth_vertices.shape[:-1], 1))],
                                    axis=-1)
        points_teeth = np.matmul(tooth_homo, PoseMat)
        mesh.vertices = o3d.utility.Vector3dVector(points_teeth)
        log_dir = "log/Export_mesh"
        transformed_stl_path = os.path.join(log_dir, f"{tooth_id}.stl")
        mesh.compute_triangle_normals()
        o3d.io.write_triangle_mesh(transformed_stl_path, mesh)

    tooth_ids_top = tooth_views.get("top", [])
    merged_mesh_top = o3d.geometry.TriangleMesh()

    for tooth_id in tooth_ids_top:
        log_dir = "log/Export_mesh"
        obj_file_path = os.path.join(log_dir, f"{tooth_id}.stl")
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        merged_mesh_top += mesh
    merged_mesh_top.remove_duplicated_vertices()
    merged_mesh_top.remove_duplicated_triangles()
    merged_mesh_top.compute_vertex_normals()
    output_stl_path = os.path.join(log_dir, "upper_transformed_model.stl")
    o3d.io.write_triangle_mesh(output_stl_path, merged_mesh_top)
    print(f"Combined STL saved to: {output_stl_path}")
    tooth_ids_bottom = tooth_views.get("bottom", [])
    merged_mesh_bottom = o3d.geometry.TriangleMesh()

    for tooth_id in tooth_ids_bottom:
        log_dir = "log/Export_mesh"
        obj_file_path = os.path.join(log_dir, f"{tooth_id}.stl")
        mesh = o3d.io.read_triangle_mesh(obj_file_path)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        merged_mesh_bottom += mesh
    merged_mesh_bottom.remove_duplicated_vertices()
    merged_mesh_bottom.remove_duplicated_triangles()
    merged_mesh_bottom.compute_vertex_normals()
    output_stl_path = os.path.join(log_dir, "lower_transformed_model.stl")
    o3d.io.write_triangle_mesh(output_stl_path, merged_mesh_bottom)
    print(f"Combined STL saved to: {output_stl_path}")
    evaluator = ToothModelEvaluator(tooth_ids, model_folder, log_dir)
    evaluator.evaluate()
