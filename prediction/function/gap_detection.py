import numpy as np
from scipy.spatial import ConvexHull
from .concave_hull import concave_hull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def is_point_in_circle(center, radius, point):
    """
    判断一个点是否在指定的圆内。

    Args:
        center: 圆心坐标，格式为 (x, y)。
        radius: 圆的半径。
        point: 需要判断的点，格式为 (x, y)。

    Returns:
        bool: 如果点在圆内，返回 True，否则返回 False。
    """
    return np.linalg.norm(np.array(center) - np.array(point)) <= radius


def has_points_in_circle(center, radius, points):
    """
    判断在指定的圆内是否有任意一个点。

    Args:
        center: 圆心坐标，格式为 (x, y)。
        radius: 圆的半径。
        points: 点集，格式为二维数组 (n, 2)，表示多个点的坐标。

    Returns:
        bool: 如果有点在圆内，返回 True，否则返回 False。
    """
    return any(is_point_in_circle(center, radius, point) for point in points)

def read_points(file_path):
    """
    从文件中读取点云数据。
    """
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    print(f"文件 {file_path} 中存在非法数据: {line.strip()}")
    if len(points) == 0:
        print(f"文件 {file_path} 无有效点云数据")
    return np.array(points)
class GapDetector:
    """
    用于检测切片中是否存在缺口的工具类。通过聚类和几何计算来判断切片中是否有缺口。
    """

    def __init__(self):
        self.cluster_info = {}

    def calculate_convex_hull_area(self, slice_points_2d):
        """
        计算给定点集的凸包面积。
        """
        if len(slice_points_2d) < 3:
            return 0
        convex_hull = ConvexHull(slice_points_2d)
        return convex_hull.volume

    def calculate_concave_hull_area(self, slice_points_2d):
        """
        计算给定点集的凹包面积。
        """
        concave_hull_points = concave_hull(slice_points_2d)
        if concave_hull_points is None or len(concave_hull_points) < 3:
            return None
        concave_hull_points = np.array(concave_hull_points)
        return 0.5 * np.abs(np.dot(concave_hull_points[:, 0], np.roll(concave_hull_points[:, 1], 1)) -
                            np.dot(concave_hull_points[:, 1], np.roll(concave_hull_points[:, 0], 1)))

    def find_longest_edge_midpoint(self, concave_hull_points):
        """
        找到凹包中最长边的中点和对应的半径。
        """
        max_length = 0
        longest_edge_midpoint = None
        for i in range(len(concave_hull_points) - 1):
            p1 = concave_hull_points[i]
            p2 = concave_hull_points[i + 1]
            edge_length = np.linalg.norm(p2 - p1)
            if edge_length > max_length:
                max_length = edge_length
                longest_edge_midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        radius = max_length * 0.25
        radius = radius if radius >= 0.5 else 0.5
        return longest_edge_midpoint, radius

    def cluster_points(self, slice_points_2d):
        """
        使用DBSCAN对点集进行聚类。
        """
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(slice_points_2d)
        return clusters

    def detect_gap_in_cluster(self, cluster_points):
        """
        判断一个聚类簇内是否存在缺口。
        """
        convex_hull_area = self.calculate_convex_hull_area(cluster_points)
        concave_hull_area = self.calculate_concave_hull_area(cluster_points)
        if concave_hull_area is None:
            return True

        if convex_hull_area / concave_hull_area > 2:
            return True

        concave_hull_points = np.array(concave_hull(cluster_points))
        if len(concave_hull_points) < 3:
            return True

        longest_edge_midpoint, radius = self.find_longest_edge_midpoint(concave_hull_points)
        if longest_edge_midpoint:
            has_point_in_circle = has_points_in_circle(longest_edge_midpoint, radius, cluster_points)
            if not has_point_in_circle:
                return True

        return False

    def visualize_cluster(self, slice_points, clusters):
        """
        可视化给定切片的簇，凸包和凹包。
        """
        unique_labels = np.unique(clusters)
        plt.figure(figsize=(8, 6))

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = slice_points[clusters == label]

            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")

            convex_hull = ConvexHull(cluster_points)
            for simplex in convex_hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')

            concave_hull_points = concave_hull(cluster_points)
            if concave_hull_points is not None and concave_hull_points.size > 0:
                concave_hull_points = np.array(concave_hull_points)
                for simplex in range(len(concave_hull_points) - 1):
                    plt.plot([concave_hull_points[simplex, 0], concave_hull_points[simplex + 1, 0]],
                             [concave_hull_points[simplex, 1], concave_hull_points[simplex + 1, 1]], 'r--')

            longest_edge_midpoint, radius = self.find_longest_edge_midpoint(concave_hull_points)
            if longest_edge_midpoint:
                radius = max(radius, 0.5)
                circle = plt.Circle(longest_edge_midpoint, radius, color='b', fill=False, linestyle='--')
                plt.gca().add_artist(circle)

                if has_points_in_circle(longest_edge_midpoint, radius, cluster_points):
                    circle.set_edgecolor('r')

        plt.title("Cluster Visualization with Convex and Concave Hulls")
        plt.legend()
        plt.show()

    def has_gap(self, slice_points_2d):
        """
        判断给定的点集（切片）中是否存在缺口。
        """
        if len(slice_points_2d) < 3:
            return True

        clusters = self.cluster_points(slice_points_2d)

        cluster_areas = {}
        for cluster_label in np.unique(clusters):

            cluster_points = slice_points_2d[clusters == cluster_label]

            cluster_area = self.calculate_convex_hull_area(cluster_points)
            cluster_areas[cluster_label] = cluster_area

        sorted_cluster_labels = sorted(cluster_areas, key=cluster_areas.get, reverse=True)

        cluster_boundary = max(cluster_areas.values()) * 0.3

        if cluster_boundary == 0:
            return True

        for cluster_label in sorted_cluster_labels:
            if cluster_areas[cluster_label] > cluster_boundary:
                cluster_points = slice_points_2d[clusters == cluster_label]
                if self.detect_gap_in_cluster(cluster_points):
                    return True

        return False


def process_point_cloud(file_path, slice_height=0.2):
    """
    读取点云文件，逐层进行切片，检查每层的缺口并进行可视化。
    """
    points = read_points(file_path)

    points = np.array(points)
    points_sorted = points[points[:, 2].argsort()]

    h = slice_height
    slice_high = points_sorted[-1][2]
    index_p = len(points_sorted) - 1
    slice_bag = []

    while index_p >= 0:
        slice_points = []
        while index_p >= 0 and points_sorted[index_p][2] >= slice_high - h:
            slice_points.append(points_sorted[index_p])
            index_p -= 1
        if slice_points:
            slice_bag.append(np.array(slice_points))
        slice_high -= h

    print(len(slice_bag))

    gap_detector = GapDetector()
    for i, slice_points in enumerate(slice_bag):
        if gap_detector.has_gap(slice_points[:, :2]):
            print(f"Layer {i + 1} has a gap!")
        else:
            print(f"Layer {i + 1} does not have a gap.")

        gap_detector.visualize_cluster(slice_points[:, :2], gap_detector.cluster_points(slice_points[:, :2]))


if __name__ == '__main__':
    file_path = r"/LSPNet/prediction/predict_result/single_sinkhole"# 替换为你的实际路径
    process_point_cloud(file_path)
