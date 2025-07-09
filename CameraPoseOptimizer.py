import numpy as np
import os
import cv2
import open3d as o3d
from lmfit import minimize
import matplotlib.pyplot as plt
from lmfit.minimizer import AbortFitException
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as RR
from joblib import Parallel, delayed


class CameraPoseOptimizer:
    def __init__(self, tooth_points_3d, contours_and_normals, teeth_pose, tooth_views, image_dimensions, camera_params,
                 iteration):
        self.tooth_points_3d = tooth_points_3d
        self.contours_and_normals = contours_and_normals
        self.tooth_views = tooth_views
        self.image_dimensions = image_dimensions
        self.camera_params = camera_params
        self.iteration = iteration
        self.max_iter = [150, 150, 150]
        self.teeth_pose = teeth_pose

    def init_edge_mask_normals(self, vertices_xy, show=False):
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

    def project_points(self, view_name, points_3d, camera_matrix, rvec, tvec, show=False):
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
        merged_contours, merged_contour_normals = self.extract_contours(visible_points_2d, view_name)
        sampled_contours = []
        sampled_normals = []
        for contour, normal in zip(merged_contours, merged_contour_normals):
            sampled_points, sampled_normals_group = self.uniform_downsample(np.array(contour), np.array(normal))
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

    def uniform_downsample(self, points, normals, num_samples=30):
        if len(points) <= num_samples:
            return points, normals
        indices = np.linspace(0, len(points) - 1, num_samples, dtype=np.int32)
        sampled_points = points[indices]
        sampled_normals = normals[indices]
        return sampled_points, sampled_normals

    def extract_contours(self, visible_points_2d, view_name):
        merged_contours = []
        merged_contour_normals = []
        for tooth_id, points_2d in visible_points_2d.items():
            if points_2d is None or len(points_2d) < 20:
                print(f"Warning: No points available for tooth {tooth_id} in view {view_name}. Skipping...")
                default_contour_points = np.ones((30, 2), dtype=np.int32)
                default_contour_normals = np.tile([0, 1], (30, 1)).astype(np.int32)
                merged_contours.append(default_contour_points)
                merged_contour_normals.append(default_contour_normals)
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
                contour = max(contours, key=cv2.contourArea)
                contour_points = contour[:, 0, :]
                contour_points[:, 0] = contour_points[:, 0] + min_x - margin
                contour_points[:, 1] = contour_points[:, 1] + min_y - margin
                contour_normals = self.init_edge_mask_normals(contour_points, False)
                merged_contours.append(contour_points)
                merged_contour_normals.append(contour_normals)
        return merged_contours, merged_contour_normals

    def contour_error(self, projected_contour, projected_normals, real_contour, real_normals, w1=0.04, w2=2, w3=0.4):
        if len(projected_contour) == 0 or len(projected_normals) == 0:
            return 0

        target_contour = np.vstack(real_contour)
        target_normals = np.vstack(real_normals)
        base_contour = np.vstack(projected_contour)
        base_normals = np.vstack(projected_normals)

        point_loss_mat = distance_matrix(base_contour, target_contour, p=2) ** 2
        normal_loss_mat = -((base_normals @ target_normals.T) ** 2) / 0.09
        loss_mat = point_loss_mat * np.exp(normal_loss_mat)

        col_indices = np.argmin(loss_mat, axis=1)
        row_indices = np.arange(len(base_contour))
        matched_base = base_contour[row_indices]
        matched_target = target_contour[col_indices]
        matched_base_normals = base_normals[row_indices]

        pp_diff = matched_target - matched_base
        num_points = len(pp_diff)
        L_pp = (np.sum(pp_diff ** 2) / num_points)

        dot_products = np.sum(pp_diff * matched_base_normals, axis=1)  # [K,]
        L_pl = (np.sum(dot_products ** 2) / num_points)  # 内积平方和

        real_centroids = np.array([np.mean(c, axis=0) for c in real_contour])  # [C,2]
        projected_centroids = np.array([np.mean(c, axis=0) for c in projected_contour])  # [C,2]

        centroid_cost = distance_matrix(projected_centroids, real_centroids, p=2)  # [C,C]
        c_row, c_col = linear_sum_assignment(centroid_cost)
        centroid_diff = real_centroids[c_col] - projected_centroids[c_row]  # [C,2]
        num_centroids = len(centroid_diff)
        L_cp = (np.sum(centroid_diff ** 2) / num_centroids)  # 平均中心点损失
        total_loss = w1 * L_pp + w2 * L_pl + w3 * L_cp
        return total_loss

    def loss_function(self, params, view_name, points_3d, real_contour, real_contour_normals):
        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']
        rvec = np.array([params['rvec_0'], params['rvec_1'], params['rvec_2']])
        tvec = np.array([params['tvec_0'], params['tvec_1'], params['tvec_2']])
        camera_matrix = np.array([[fx, 0, 0], [0, fy, 0], [cx, cy, 1]])
        projected_contour, projected_contour_normal = self.project_points(view_name, points_3d, camera_matrix, rvec,
                                                                          tvec)
        return self.contour_error(projected_contour, projected_contour_normal, real_contour, real_contour_normals)

    def optimize_view(self, view_name):
        tooth_ids = self.tooth_views.get(view_name, [])
        points_3d = {}
        for tooth_id in tooth_ids:
            points_3d[tooth_id] = self.tooth_points_3d[tooth_id]
        real_contour = self.contours_and_normals[view_name]['contour']
        real_normals = self.contours_and_normals[view_name]['normals']
        params = self.camera_params[view_name]
        log_dir = "log/CameraPoseOptimizer"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{self.iteration}-{view_name}_optimization.txt")
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        tolerance_residual = 1e-3
        tolerance_param_change = 1e-4
        prev_residual = None
        prev_params = None
        stop_flag = False
        if self.iteration < len(self.max_iter):
            maxiter = self.max_iter[self.iteration]
        else:
            maxiter = self.max_iter[-1]

        def print_info(params, iter, residuals, *args, **kws):
            nonlocal prev_residual, prev_params, stop_flag
            if stop_flag:
                return

            with open(log_file_path, 'a') as f:
                f.write(f"Iteration {iter}: ")
                for name, value in params.items():
                    f.write(f"{name}={value.value:.6f} ")
                if residuals is not None:
                    if np.isscalar(residuals):
                        current_residual = residuals
                        f.write(f"Residual: {current_residual:.6f}")
                    else:
                        current_residual = np.mean(residuals)
                        f.write(f"Residuals (mean): {current_residual:.6f} (first 5): {residuals[:5]}")
                f.write("\n")

            if prev_residual is not None or prev_params is not None:
                residual_change = abs(prev_residual - current_residual)
                param_change = max(abs(prev_params[name] - p.value) for name, p in params.items())

                if residual_change < tolerance_residual or param_change < tolerance_param_change:
                    print(f"Optimization stopping early at iteration {iter}.")
                    stop_flag = True
                    return True
            prev_residual = current_residual
            prev_params = {name: p.value for name, p in params.items()}

        try:
            result = minimize(
                self.loss_function,
                params,
                args=(view_name, points_3d, real_contour, real_normals),
                method='nelder',
                max_nfev=maxiter,
                iter_cb=print_info
            )
            return result.params
        except AbortFitException:
            print(f"Optimization for view {view_name} stopped early.")
            return params

    def optimize_all_views(self):
        """优化所有视图的相机姿态."""
        for tooth_id in self.teeth_pose.keys():
            params = self.teeth_pose[tooth_id]
            tooth_rvec = np.array([params['rvec_0'], params['rvec_1'], params['rvec_2']])
            tooth_tvec = np.array([params['tvec_0'], params['tvec_1'], params['tvec_2']])
            current_tooth_vertices = self.tooth_points_3d[tooth_id]
            Rvec = RR.from_rotvec(tooth_rvec).as_matrix().T
            PoseMat = np.vstack([Rvec, tooth_tvec])
            tooth_homo = np.concatenate([current_tooth_vertices, np.ones((*current_tooth_vertices.shape[:-1], 1))],
                                        axis=-1)
            points_teeth = np.matmul(tooth_homo, PoseMat)
            self.tooth_points_3d[tooth_id] = points_teeth
        view_names = list(self.tooth_views.keys())
        results = Parallel(n_jobs=-1)(
            delayed(self.optimize_view)(view_name) for view_name in view_names
        )
        optimized_params = dict(zip(view_names, results))
        return optimized_params
