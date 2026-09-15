import numpy as np
import alphashape
from shapely.geometry import Polygon

def concave_hull(points, concavity=2.0, length_threshold=1.0):
    """
    计算点集的凹包，使用 alphashape（一个成熟稳定的凹包算法）
    参数：
        points: Nx2 或 Nx3 点集（将自动取前两列）
        concavity: 控制凹包曲折程度（越小越贴合点云）
        length_threshold: 控制边界的平滑程度
    返回：
        hull_points: 凹包点集 Nx2
    """

    pts = np.array(points)
    if pts.shape[1] > 2:
        pts = pts[:, :2]

    if len(pts) < 4:
        return pts

    alpha_shape = alphashape.alphashape(pts, concavity)

    if isinstance(alpha_shape, Polygon):
        hull = np.array(alpha_shape.exterior.coords)
        return hull

    return pts
