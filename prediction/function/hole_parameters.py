from shapely.geometry import Polygon
from .concave_hull import concave_hull
from typing import Dict, Tuple, Optional, List
import numpy as np


class ConvexHullProperties:
    """凸包属性计算类，负责凸包相关几何参数的计算"""
    import numpy as np

    def __init__(self, points: np.ndarray):
        self.points = np.array(points)
        self.hull_points = None
        self.area = None
        self.perimeter = None
        self.centroid = None

    def compute(self):
        """计算凸包所有属性"""
        if self.points is None or len(self.points) < 3:
            self.area = 0
            self.perimeter = 0
            self.centroid = 0, 0
            return
        self._compute_hull_points()
        self._compute_area_and_perimeter()
        self._compute_centroid()

    def _compute_hull_points(self):
        """使用Graham扫描算法计算凸包点集"""
        sorted_points = sorted(self.points, key=lambda p: (p[0], p[1]))

        def cross(o, a, b):
            """计算叉积判断三点方向"""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        self.hull_points = np.array(lower[:-1] + upper[:-1])

    def _compute_area_and_perimeter(self):
        """计算凸包面积和周长"""
        if self.hull_points is None or len(self.hull_points) < 3:
            return
        polygon = Polygon(self.hull_points)
        self.area = polygon.area
        self.perimeter = polygon.length

    def _compute_centroid(self):
        """计算凸包质心"""
        if self.hull_points is not None:
            self.centroid = np.mean(self.hull_points, axis=0)

    def get_properties(self) -> Dict:
        """返回凸包所有属性"""
        return {
            'hull_points': self.hull_points,
            'area': self.area,
            'perimeter': self.perimeter,
            'centroid': self.centroid
        }


class ConcaveHullProperties:
    """凹包属性计算类，负责凹包相关几何参数的计算"""

    def __init__(self, points: np.ndarray):
        self.points = np.array(points)
        self.hull_points = None
        self.area = None
        self.perimeter = None

    def compute(self):
        """计算凹包所有属性"""
        self._compute_hull_points()
        self._compute_area_and_perimeter()

    def _compute_hull_points(self):
        """使用concave_hull库计算凹包点集"""
        if self.points is None or len(self.points) < 3:
            self.area = 0
            self.perimeter = 0
            return

        self.hull_points = concave_hull(self.points, length_threshold=1, concavity=2.5)

    def _compute_area_and_perimeter(self):
        """计算凹包面积和周长"""
        if self.hull_points is None or len(self.hull_points) < 3:
            self.area = 0
            self.perimeter = 0
            return
        polygon = Polygon(self.hull_points)
        self.area = polygon.area
        self.perimeter = polygon.length

    def get_properties(self) -> Dict:
        """返回凹包所有属性"""
        return {
            'hull_points': self.hull_points,
            'area': self.area,
            'perimeter': self.perimeter
        }


#
class MinimumBoundingRectangle:
    """最小外接矩形计算类，负责矩形相关参数的计算"""

    def __init__(self, points: np.ndarray):
        self.points = np.array(points)
        self.long_axis = None
        self.short_axis = None
        self.azimuth = None
        self.vertices = None
        self.area = None
        self.perimeter = None
        self.eccentricity = None

        self.convex_hull_points = None

    def compute(self, convex_hull_points: Optional[np.ndarray] = None):
        """
        优化版本：使用双指针旋转卡壳法，寻找短轴最短的外接矩形（y轴基准方位角）
        时间复杂度：O(n)，n为凸包顶点数
        """
        hull_points = convex_hull_points if convex_hull_points is not None else self.points
        self.convex_hull_points = hull_points
        n = len(hull_points)

        if n < 3:
            self.long_axis = 0.0
            self.short_axis = 0.0
            self.azimuth = 0.0
            return

        min_short_axis = np.inf
        j = 1
        best_theta = 0.0

        for i in range(n):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)

            if edge_len < 1e-6:
                continue

            theta_y = np.arctan2(edge[0], edge[1])

            normal = np.array([-edge[1], edge[0]])
            normal_unit = normal / edge_len

            while True:
                j_next = (j + 1) % n
                dist_j = np.dot(hull_points[j] - p1, normal_unit)
                dist_j_next = np.dot(hull_points[j_next] - p1, normal_unit)

                if dist_j_next > dist_j + 1e-8:
                    j = j_next
                else:
                    break

            current_short_candidate = np.dot(hull_points[j] - p1, normal_unit)
            edge_unit = edge / edge_len
            projections = np.dot(hull_points - p1, edge_unit)
            current_long_candidate = projections.max() - projections.min()

            if current_short_candidate < min_short_axis:
                min_short_axis = current_short_candidate
                self.short_axis = current_short_candidate
                self.long_axis = current_long_candidate
                best_theta = theta_y

        if self.short_axis > self.long_axis:
            self.short_axis, self.long_axis = self.long_axis, self.short_axis
            self.azimuth = np.degrees(best_theta + np.pi / 2) % 180
        else:
            self.azimuth = np.degrees(best_theta) % 180

    def compute_vertice(self):
        """计算矩形四个顶点坐标（严格匹配y轴基准方位角）"""
        if self.long_axis is None or self.short_axis is None or self.azimuth is None:
            return

        theta = np.radians(self.azimuth)
        dir_x = np.sin(theta)
        dir_y = np.cos(theta)
        long_axis_unit = np.array([dir_x, dir_y])
        short_axis_unit = np.array([dir_y, -dir_x])

        target_points = self.convex_hull_points if self.convex_hull_points is not None else self.points
        long_projections = np.dot(target_points, long_axis_unit)
        min_long = long_projections.min()
        max_long = long_projections.max()
        short_projections = np.dot(target_points, short_axis_unit)
        min_short = short_projections.min()
        max_short = short_projections.max()

        self.vertices = [
            (
                min_long * long_axis_unit[0] + min_short * short_axis_unit[0],
                min_long * long_axis_unit[1] + min_short * short_axis_unit[1]
            ),
            (
                max_long * long_axis_unit[0] + min_short * short_axis_unit[0],
                max_long * long_axis_unit[1] + min_short * short_axis_unit[1]
            ),
            (
                max_long * long_axis_unit[0] + max_short * short_axis_unit[0],
                max_long * long_axis_unit[1] + max_short * short_axis_unit[1]
            ),
            (
                min_long * long_axis_unit[0] + max_short * short_axis_unit[0],
                min_long * long_axis_unit[1] + max_short * short_axis_unit[1]
            )
        ]

    def compute_metrics(self):
        """计算矩形面积、周长和偏心率"""
        if self.long_axis and self.short_axis:
            self.area = self.long_axis * self.short_axis
            self.perimeter = 2 * (self.long_axis + self.short_axis)

    def get_properties(self) -> Dict:
        """返回最小外接矩形所有属性"""
        return {
            'long_axis': self.long_axis,
            'short_axis': self.short_axis,
            'azimuth': self.azimuth,
            'vertices': self.vertices,
            'area': self.area,
            'perimeter': self.perimeter,
            'eccentricity': self.eccentricity
        }


from typing import Dict, Optional

################################################################################
import random
import math


def distance(p1, p2):
    """计算两点间的欧几里得距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def make_circle(p1, p2):
    """生成经过两点的最小圆（以两点连线为直径）
    返回值：(圆心x, 圆心y, 半径)
    """
    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2
    r = distance(p1, p2) / 2
    return (cx, cy, r)


def make_circle_from_three(p1, p2, p3):
    """生成经过三点的圆（或处理退化情况）
    返回值：(圆心x, 圆心y, 半径)
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(d) < 1e-12:
        pairs = [(p1, p2), (p1, p3), (p2, p3)]
        (u, v) = max(pairs, key=lambda t: distance(t[0], t[1]))
        return make_circle(u, v)

    ux = ((ax ** 2 + ay ** 2) * (by - cy) +
          (bx ** 2 + by ** 2) * (cy - ay) +
          (cx ** 2 + cy ** 2) * (ay - by)) / d

    uy = ((ax ** 2 + ay ** 2) * (cx - bx) +
          (bx ** 2 + by ** 2) * (ax - cx) +
          (cx ** 2 + cy ** 2) * (bx - ax)) / d

    r = distance((ux, uy), p1)
    return (ux, uy, r)


def is_point_inside_circle(p, circle):
    """判断点是否在圆内（含边界）"""
    return distance(p, (circle[0], circle[1])) <= circle[2] + 1e-12  # 容差处理浮点误差

# ---------- Welzl算法：求解最小包围圆 ----------

def welzl(points, boundary=None):
    """递归实现Welzl算法
    points: 待处理的点集
    boundary: 边界点集（已确定在最小包围圆上的点）
    返回值：(圆心x, 圆心y, 半径)
    """
    boundary = boundary or []

    if not points or len(boundary) == 3:
        if len(boundary) == 0: return (0, 0, 0)
        if len(boundary) == 1: return (boundary[0][0], boundary[0][1], 0)
        if len(boundary) == 2: return make_circle(*boundary)
        return make_circle_from_three(*boundary)

    p = points.pop()

    circle = welzl(points, boundary)

    if is_point_inside_circle(p, circle):
        points.append(p)
        return circle

    boundary.append(p)
    circle = welzl(points, boundary)
    boundary.pop()
    points.append(p)
    return circle


# ---------- 最小包围圆类 ----------


class MinimumBoundingCircle:
    """最小外接圆计算类，负责数据交换和结果存储"""

    def __init__(self, points: np.ndarray):
        """
        初始化最小外接圆计算器
        参数:
            points: 输入点集，形状为(n, 2)的NumPy数组
        """
        self.points = points.tolist()
        self.radius = None
        self.center = None
        self.area = None

    def min_enclosing_circle(self):
        """Welzl算法包装函数（随机化以提高效率）"""
        pts = self.points[:]
        random.shuffle(pts)
        return welzl(pts)

    def compute(self, points):
        """计算最小外接圆属性"""
        self.points = points.tolist()

        if len(self.points) == 0:
            raise ValueError("输入点集不能为空")

        a = self.min_enclosing_circle()
        self.center, self.radius = (a[0], a[1]), a[2]

        self._compute_circle_metrics()

    def _compute_circle_metrics(self):
        """计算圆的相关属性（面积）"""
        if self.radius is not None:
            self.area = math.pi * (self.radius ** 2)

    def get_properties(self):
        """返回最小外接圆的所有属性"""
        return {
            'radius': self.radius,
            'center': self.center,
            'area': self.area
        }


class hole_parameters:
    """点云分析主类，整合所有几何属性计算"""

    def __init__(self, points_3d: np.ndarray):
        points_2d = points_3d[:, :2]

        self.points = np.array(points_2d)
        self.convex_hull = ConvexHullProperties(points_2d)
        self.concave_hull = ConcaveHullProperties(points_2d)
        self.bounding_rect = MinimumBoundingRectangle(points_2d)
        self.bounding_circle = MinimumBoundingCircle(points_2d)

    def analyze(self):
        """执行完整的点云几何属性分析"""
        self.convex_hull.compute()
        self.concave_hull.compute()
        self.bounding_rect.compute(self.convex_hull.hull_points)
        self.bounding_circle.compute(self.convex_hull.hull_points)

        return self

    def get_all_properties(self) -> Dict:
        """返回所有几何属性的整合结果"""
        return {
            'convex_hull': self.convex_hull.get_properties(),
            'concave_hull': self.concave_hull.get_properties(),
            'bounding_rectangle': self.bounding_rect.get_properties(),
            'bounding_circle': self.bounding_circle.get_properties()
        }


# ------------------- 绘图函数 -------------------
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_concave_hull(points: np.ndarray, concave_hull_properties):
    """
    绘制凹包绕边情况并打印相关属性
    :param points: 原始点云数据
    :param concave_hull_properties: 凹包属性计算类实例
    """
    plt.scatter(points[:, 0], points[:, 1], s=5, label='原始点云')

    hull_points = concave_hull_properties.hull_points
    if hull_points is not None:
        hull_points = np.vstack((hull_points, hull_points[0]))  # 闭合凹包
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='凹包')

    plt.legend()
    plt.title('凹包绕边情况')
    plt.axis('equal')

    print("凹包属性：")
    print(f"面积: {concave_hull_properties.area:.3f}")
    print(f"周长: {concave_hull_properties.perimeter:.3f}")
    if hull_points is not None:
        centroid = np.mean(hull_points, axis=0)
        print(f"中心点: {centroid}")

    plt.show()


def plot_bounding_shapes(points: np.ndarray, bounding_rect, bounding_circle):
    """
    绘制最小外接矩形、最小外接圆及方位角方向向量
    适配使用原生Python元组存储的顶点坐标
    :param points: 原始点云数据
    :param bounding_rect: 最小外接矩形计算类实例
    :param bounding_circle: 最小外接圆计算类实例
    """

    bounding_rect.compute_vertice()
    bounding_rect.compute_metrics()

    plt.scatter(points[:, 0], points[:, 1], s=5, label='原始点云')

    if bounding_rect.vertices is not None:
        rect_vertices_np = np.array(bounding_rect.vertices)
        rect_points = np.vstack((rect_vertices_np, rect_vertices_np[0]))
        plt.plot(rect_points[:, 0], rect_points[:, 1], 'g--', label='最小外接矩形')

    if bounding_rect.vertices is not None and bounding_circle.center is not None and bounding_circle.radius is not None:
        circle = plt.Circle(
            bounding_circle.center,
            bounding_circle.radius,
            color='b',
            fill=False,
            linestyle='--',
            label='最小外接圆'
        )
        plt.gca().add_patch(circle)

    if (bounding_rect.azimuth is not None and
            bounding_rect.long_axis is not None and
            bounding_rect.vertices is not None):

        rect_vertices_np = np.array(bounding_rect.vertices)
        rect_center = np.mean(rect_vertices_np, axis=0)

        angle_rad = np.radians(bounding_rect.azimuth)
        vec_length = bounding_rect.long_axis / 2
        dir_x = np.sin(angle_rad) * vec_length
        dir_y = np.cos(angle_rad) * vec_length

        plt.quiver(
            rect_center[0], rect_center[1],
            dir_x, dir_y,
            color='red',
            scale=1,
            scale_units='xy',
            width=0.002,
            headwidth=8,
            label='方位角方向'
        )

    plt.legend()
    plt.title('最小外接矩形、最小外接圆及方位角方向')
    plt.axis('equal')

    print("\n最小外接矩形属性：")
    print(f"长轴: {bounding_rect.long_axis:.3f}" if bounding_rect.long_axis is not None else "长轴: 未计算")
    print(f"短轴: {bounding_rect.short_axis:.3f}" if bounding_rect.short_axis is not None else "短轴: 未计算")
    print(f"方位角: {bounding_rect.azimuth:.2f}°" if bounding_rect.azimuth is not None else "方位角: 未计算")
    print(f"矩形顶点:\n{bounding_rect.vertices}")
    print(f"面积: {bounding_rect.area:.3f}" if bounding_rect.area is not None else "面积: 未计算")
    print(f"周长: {bounding_rect.perimeter:.3f}" if bounding_rect.perimeter is not None else "周长: 未计算")
    print("\n最小外接圆属性：")
    print(f"半径: {bounding_circle.radius:.3f}" if bounding_circle.radius is not None else "半径: 未计算")
    print(f"面积: {bounding_circle.area:.3f}" if bounding_circle.area is not None else "面积: 未计算")
    print(f"圆心: {bounding_circle.center}" if bounding_circle.center is not None else "圆心: 未计算")

    plt.show()


if __name__ == '__main__':

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 100

    analyzer = hole_parameters(points).analyze()
    plot_concave_hull(points, analyzer.concave_hull)
    plot_bounding_shapes(points, analyzer.bounding_rect, analyzer.bounding_circle)
