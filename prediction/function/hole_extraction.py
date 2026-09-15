import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from .gap_detection import GapDetector, read_points
from .hole_parameters import hole_parameters, plot_concave_hull, plot_bounding_shapes

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class hole_extraction:
    def __init__(self, slice_thickness=0.5):
        """
        初始化洞口一次截取处理器。

        Args:
            slice_thickness (float, optional): 切片厚度。默认为0.5。
        """
        self.slice_thickness = slice_thickness
        self.critical_h = None

    def slice_point_cloud(self, points):
        """
        对点云进行切片。

        Args:
            points (np.ndarray): 点云数据，形状为(n, 3)。

        Returns:
            list: 切片后的点云列表，每个元素是一个切片的点云数据。
        """
        if len(points) == 0:
            return []

        points_sorted = points[points[:, 2].argsort()[::-1]]
        slice_high = points_sorted[0][2]
        index_p = 0
        slice_bag = []

        while index_p < len(points_sorted):
            slice_points = []
            while index_p < len(points_sorted) and points_sorted[index_p][2] >= slice_high - self.slice_thickness:
                slice_points.append(points_sorted[index_p])
                index_p += 1
            if slice_points:
                slice_bag.append(slice_points)
            slice_high -= self.slice_thickness

        return slice_bag

    def compute_area_list(self, slice_bag):
        """
        计算每个切片的凸包面积和最低高度。

        Args:
            slice_bag (list): 切片后的点云列表。

        Returns:
            list: 每个切片的凸包面积和最低高度列表。
        """
        area_list = []

        for bag in slice_bag:
            if len(bag) < 3:
                min_height = np.min([p[2] for p in bag])
                area_list.append([0, min_height])
                continue

            try:
                xy_points = [(p[0], p[1]) for p in bag]
                hull = ConvexHull(xy_points)
                min_height = np.min([p[2] for p in bag])
                area_list.append([hull.volume, min_height])
            except:
                min_height = np.min([p[2] for p in bag])
                area_list.append([0, min_height])

        return area_list

    def find_critical_height(self, area_list, slice_bag):
        """
        找到临界高度。
        """
        if not area_list:
            return

        for i in range(1, len(area_list)):
            delta_area = area_list[i][0] - area_list[i - 1][0]
            points = np.array(slice_bag[i])

            has_gap = GapDetector().has_gap(points[:, :2])

            if delta_area < 0 and not has_gap:
                self.critical_h = area_list[i][1]
                return

    def crop_point_cloud(self, points):
        """
        按照临界高度裁剪点云。

        Args:
            points (np.ndarray): 点云数据，形状为(n, 3)。

        Returns:
            np.ndarray: 裁剪后的点云数据。
        """
        if self.critical_h is not None:
            return points[points[:, 2] >= self.critical_h]
        return points.copy()

    def process(self, points):
        """
        处理点云数据并返回裁剪后的点云。

        Args:
            points (np.ndarray): 点云数据，形状为(n, 3)。

        Returns:
            np.ndarray: 裁剪后的点云数据。
        """
        if len(points) == 0:
            return np.array([])

        slice_bag = self.slice_point_cloud(points)
        area_list = self.compute_area_list(slice_bag)
        self.find_critical_height(area_list, slice_bag)
        cropped_points = self.crop_point_cloud(points)

        return cropped_points, area_list

    def visualize_original(self, original_points):
        """
        可视化原始点云和裁剪面，优化了视觉效果。

        Args:
            original_points (np.ndarray): 原始点云数据，形状为(N, 3)
        """

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            original_points[:, 0], original_points[:, 1], original_points[:, 2],
            c=original_points[:, 2],
            cmap='viridis',
            s=10,
            alpha=0.7,
            edgecolors='none',
            label='原始点云'
        )

        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('高度值', rotation=270, labelpad=15)

        if self.critical_h is not None:
            x_range = np.linspace(original_points[:, 0].min(), original_points[:, 0].max(), 50)
            y_range = np.linspace(original_points[:, 1].min(), original_points[:, 1].max(), 50)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, self.critical_h)

            ax.plot_surface(
                X, Y, Z,
                color='red',
                alpha=0.3,
                label=f'临界高度: {self.critical_h:.2f}'
            )

        ax.set_title("原始点云与临界高度平面", fontsize=15, pad=20)
        ax.set_xlabel("X轴", fontsize=12, labelpad=10)
        ax.set_ylabel("Y轴", fontsize=12, labelpad=10)
        ax.set_zlabel("Z轴", fontsize=12, labelpad=10)

        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.view_init(elev=30, azim=45)

        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        plt.show()

    def visualize_cropped(self, cropped_points):
        """
        可视化裁剪后的点云。

        Args:
            cropped_points (np.ndarray): 裁剪后的点云数据。
        """
        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cropped_points[:, 0], cropped_points[:, 1], cropped_points[:, 2], c='g', label='Cropped Points')

        ax.set_title("Cropped Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        plt.show()

    def visualize_area_vs_height(self, area_list):
        """
        可视化面积与高度的关系（折线图）。

        Args:
            area_list (list): 包含每个切片的面积和最低高度的列表。
        """
        heights = [item[1] for item in area_list]
        areas = [item[0] for item in area_list]

        plt.figure(figsize=(8, 6))
        plt.plot(heights, areas, marker='o', color='b', label='Area vs Height')
        plt.title("Area vs Height")
        plt.xlabel("Height")
        plt.ylabel("Convex Hull Area")
        plt.legend()
        plt.grid(True)

        plt.show()


if __name__ == "__main__":
    file_path = r"/LSPNet/prediction/predict_result/single_sinkhole"# 替换为你的实际路径

    points = read_points(file_path)
    points = np.array(points)
    processor = hole_extraction(slice_thickness=0.5)
    cropped_points, area_list = processor.process(points)
    print("裁剪后的点云数据:")
    processor.visualize_original(points)
    processor.visualize_cropped(cropped_points)
    processor.visualize_area_vs_height(area_list)
    analyzer = hole_parameters(cropped_points[:, :2]).analyze()
    plot_concave_hull(points, analyzer.concave_hull)
    plot_bounding_shapes(cropped_points, analyzer.bounding_rect, analyzer.bounding_circle)
