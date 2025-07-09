import os
import numpy as np
import open3d as o3d


class ToothModelEvaluator:
    def __init__(self, tooth_ids, source_dir, target_dir):
        self.tooth_ids = tooth_ids
        self.source_dir = source_dir
        self.target_dir = target_dir

        self.vertex_errors = []
        self.rotation_errors = []
        self.translation_errors = []

    def load_point_cloud_from_stl(self, stl_path):
        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        return pcd

    def compute_AVD(self, pcd_pred, pcd_gt):
        pcd_gt_tree = o3d.geometry.KDTreeFlann(pcd_gt)
        pred_points = np.asarray(pcd_pred.points)
        gt_points = np.asarray(pcd_gt.points)
        distances = []

        for point in pred_points:
            [_, idx, _] = pcd_gt_tree.search_knn_vector_3d(point, 1)
            nearest_point = gt_points[idx[0]]
            dist = np.linalg.norm(point - nearest_point)
            distances.append(dist)

        return np.mean(distances)

    def compute_icp_transform(self, source, target):
        threshold = 100.0
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        return icp_result.transformation

    def rotation_error_from_matrix(self, T):
        R = T[:3, :3]
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def evaluate(self):
        print("每颗牙齿的误差（单位：mm / 度）")
        print("ToothID\tVertexError\tRotationError\tTranslationError")

        for tid in self.tooth_ids:
            src_file = os.path.join(self.source_dir, f"{tid}.stl")
            tgt_file = os.path.join(self.target_dir, f"{tid}.stl")

            if not os.path.exists(src_file) or not os.path.exists(tgt_file):
                print(f"{tid}\t缺失模型，跳过")
                continue

            src_pcd = self.load_point_cloud_from_stl(src_file)
            tgt_pcd = self.load_point_cloud_from_stl(tgt_file)

            v_error = self.compute_AVD(src_pcd, tgt_pcd)
            T = self.compute_icp_transform(src_pcd, tgt_pcd)
            r_error = self.rotation_error_from_matrix(T)
            t_error = np.linalg.norm(T[:3, 3])

            self.vertex_errors.append(v_error)
            self.rotation_errors.append(r_error)
            self.translation_errors.append(t_error)

            print(f"{tid}\t{v_error:.4f}\t\t{r_error:.4f}\t\t{t_error:.4f}")

        print("\n平均误差：")
        print(f"平均顶点误差: {np.mean(self.vertex_errors):.4f} mm")
        print(f"平均旋转误差: {np.mean(self.rotation_errors):.4f} °")
        print(f"平均平移误差: {np.mean(self.translation_errors):.4f} mm")
