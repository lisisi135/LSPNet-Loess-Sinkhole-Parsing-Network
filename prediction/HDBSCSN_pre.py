import numpy as np
import os
from sklearn.cluster import DBSCAN
import hdbscan
import cv2

prediction_file = r"/best_prediction.txt"# 替换为你的实际路径
output_dir = r"/LSNet/prediction/predict_result"# 替换为你的实际路径
single_sinkhole_folder = os.path.join(output_dir, "single_sinkhole")
os.makedirs(single_sinkhole_folder, exist_ok=True)

print(f"正在加载预测结果：{prediction_file}")
data = np.loadtxt(prediction_file)

sinkhole = data[data[:, 3] == 1]
print(f"检测到 {sinkhole.shape[0]} 个 sinkhole 点")

if sinkhole.shape[0] == 0:
    print("没有检测到 sinkhole 点，结束处理。")
else:
    print("开始聚类（HDBSCAN）...")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, cluster_selection_method='eom')
    labels = clusterer.fit_predict(sinkhole[:, :3])
    unique_labels = np.unique(labels)

    print(f"HDBSCAN 发现 {len(unique_labels) - (1 if -1 in labels else 0)} 个落水洞（不包括噪声）")

    count = 0
    for label in unique_labels:
        if label == -1:
            print("跳过噪声点")
            continue
        cluster_points = sinkhole[labels == label]
        print(f"落水洞 {count}，点数: {len(cluster_points)}")

        if len(cluster_points) > 50:
            file_name = os.path.join(single_sinkhole_folder, f"sinkhole_{count}.txt")
            np.savetxt(file_name, cluster_points)
            print(f"已保存 {file_name}")
            count += 1
        else:
            print(f"落水洞 {count} 点数过少（{len(cluster_points)}），跳过")

    np.savetxt(os.path.join(output_dir, "sinkhole_all.txt"), sinkhole)
    np.savetxt(os.path.join(output_dir, "sinkhole+non_sinkhole.txt"), data)
    print("全部保存完成")