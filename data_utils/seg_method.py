import os
import numpy as np
import random

def seg80000(sinkhole_samples, sinkhole_labels, whole_place, file_path):
    cut_direction = determineCutDirection(whole_place)
    point_idx = np.argsort(whole_place[:, cut_direction])
    point_idx_reverse = np.argsort(whole_place[:, cut_direction])[::-1]

    if os.path.isfile(file_path):
        # 将数据切分为80000个点一个块
        for i in range(int(whole_place.shape[0] / 80000)):
            save_points = whole_place[point_idx][i * 80000:(i + 1) * 80000, :]
            cut_direction_block = determineCutDirection(save_points)
            per_block_index = np.argsort(save_points[:, cut_direction_block])
            per_block_index_reverse = np.argsort(save_points[:, cut_direction_block])[::-1]

            # 对每个块再次进行16000点的切分
            for j in range(int(save_points.shape[0] / 16000)):
                per_block, per_label = save_points[per_block_index][j * 16000:(j + 1) * 16000, 0:3], save_points[per_block_index][j * 16000:(j + 1) * 16000, -1]
                sinkhole_samples.append(per_block)
                sinkhole_labels.append(per_label)

            per_block_reverse, per_label_reverse = save_points[per_block_index_reverse][0:16000, 0:3], save_points[per_block_index_reverse][0:16000, -1]
            sinkhole_samples.append(per_block_reverse)
            sinkhole_labels.append(per_label_reverse)

        # 处理最后一块的数据
        save_points_last = whole_place[point_idx_reverse][0:80000, :]
        cut_direction_lastblock = determineCutDirection(save_points_last)
        per_block_last_index = np.argsort(save_points_last[:, cut_direction_lastblock])
        per_block_last_index_reverse = np.argsort(save_points_last[:, cut_direction_lastblock])[::-1]

        # 将最后一块也进行16000点的切割
        for j in range(int(save_points_last.shape[0] / 16000)):
            per_block, per_label = save_points_last[per_block_last_index][j * 16000:(j + 1) * 16000, 0:3], save_points_last[per_block_last_index][j * 16000:(j + 1) * 16000, -1]
            sinkhole_samples.append(per_block)
            sinkhole_labels.append(per_label)

        per_block_reverse, per_label_reverse = save_points_last[per_block_last_index_reverse][0:16000, 0:3], save_points_last[per_block_last_index_reverse][0:16000, -1]
        sinkhole_samples.append(per_block_reverse)
        sinkhole_labels.append(per_label_reverse)

def seg16000(sinkhole_samples, sinkhole_labels, whole_place, file_path):
    cut_direction = determineCutDirection(whole_place)
    point_idx = np.argsort(whole_place[:, cut_direction])
    point_idx_reverse = np.argsort(whole_place[:, cut_direction])[::-1]

    if os.path.isfile(file_path):
        for i in range(int(whole_place.shape[0] / 16000)):
            per_block, per_label = whole_place[point_idx][i * 16000:(i + 1) * 16000, 0:3], whole_place[point_idx][i * 16000:(i + 1) * 16000, -1]
            sinkhole_samples.append(per_block)
            sinkhole_labels.append(per_label)

    per_block_last, per_label_last = whole_place[point_idx_reverse][0:16000, 0:3], whole_place[point_idx_reverse][0:16000, -1]
    sinkhole_samples.append(per_block_last)
    sinkhole_labels.append(per_label_last)

def determineCutDirection(whole_place):
    """
    根据点云数据在x、y、z维度上的分布范围，决定切割方向。
    返回值：0 - x维度，1 - y维度，2 - z维度
    """
    x_range = whole_place[:, 0].max() - whole_place[:, 0].min()
    y_range = whole_place[:, 1].max() - whole_place[:, 1].min()
    z_range = whole_place[:, 2].max() - whole_place[:, 2].min()

    if x_range >= y_range and x_range >= z_range:
        return 0
    elif y_range >= x_range and y_range >= z_range:
        return 1
    else:
        return 2

def determineCutCount(whole_place):
    point_idx = np.argsort(whole_place[:, determineCutDirection(whole_place)])
    part_point = whole_place[point_idx][0:int(point_idx.size / 20)]
    x_range = part_point[:, 0].max() - part_point[:, 0].min()
    y_range = whole_place[:, 1].max() - whole_place[:, 1].min()

    if x_range / y_range > 10 or y_range / x_range > 10:
        return 0
    else:
        return 1

def sampling_percentage(points, labels, transform):
    random_sample_rate = np.random.uniform(0.5, 1.0)
    num_samples = int(points.shape[0] * random_sample_rate)

    indices = np.random.choice(points.shape[0], size=num_samples, replace=False)
    sampled_points = points[indices, :]
    sampled_labels = labels[indices]

    target_num_points = 16000
    if sampled_points.shape[0] > target_num_points:
        indices = np.random.choice(sampled_points.shape[0], size=target_num_points, replace=False)
        sampled_points = sampled_points[indices, :]
        sampled_labels = sampled_labels[indices]
    elif sampled_points.shape[0] < target_num_points:
        num_fill_points = target_num_points - sampled_points.shape[0]
        fill_indices = np.random.choice(sampled_points.shape[0], size=num_fill_points, replace=True)
        fill_points = sampled_points[fill_indices, :]
        fill_labels = sampled_labels[fill_indices]
        sampled_points = np.vstack((sampled_points, fill_points))
        sampled_labels = np.concatenate((sampled_labels, fill_labels))

    if transform:
        if transform == 'translation':
            sampled_points = random_translation(sampled_points)
        elif transform == 'rotation':
            sampled_points = random_rotation(sampled_points)
        elif transform == 'scaling':
            sampled_points = random_scaling(sampled_points)
        elif transform == 'noise':
            sampled_points = add_noise(sampled_points)

    return sampled_points, sampled_labels

def random_translation(points, max_translation=0.1):
    """随机平移变换"""
    translation = np.random.uniform(-max_translation, max_translation, size=(1, 3))
    return points + translation

def random_rotation(points):
    """随机旋转变换"""
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    points = points @ R
    return points

def random_scaling(points, scale_range=(0.8, 1.2)):
    """随机缩放变换"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return points * scale

def add_noise(points, noise_level=0.01):
    """添加随机噪声"""
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise

if __name__ == "__main__":
    data_root = r"/LSPNet/data/testing_data"
    filelist = sorted(os.listdir(data_root))
    files = [file for file in filelist if ".txt" in file]
    sinkhole_samples = []
    sinkhole_labels = []
    seg80000(sinkhole_samples, sinkhole_labels, data_root, files)
