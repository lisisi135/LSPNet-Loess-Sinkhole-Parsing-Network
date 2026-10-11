from data_utils.seg_method import *
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class cityEnvDataLoader(Dataset):
    def __init__(self, split='train', data_root=r'', num_point=16000, sample_rate=1.0, transform=None, use_hard_sampler=True, no_sample=False):
        super().__init__()
        self.num_point = num_point
        self.sample_rate = sample_rate
        self.transform = transform
        self.use_hard_sampler = use_hard_sampler
        self.no_sample = no_sample
        filelist = sorted(os.listdir(data_root))
        files = [file for file in filelist if ".txt" in file]
        self.sinkhole_samples = []
        self.sinkhole_labels = []

        for file in files:
            file_path = os.path.join(data_root, file)
            whole_place = np.loadtxt(file_path)[:, [0, 1, 2, -1]].astype(np.float32)
            if determineCutDirection(whole_place) == 0:
                seg16000(self.sinkhole_samples, self.sinkhole_labels, whole_place, file_path)
            else:
                seg80000(self.sinkhole_samples, self.sinkhole_labels, whole_place, file_path)

        self.sinkhole_patch_idxs = [i for i, label in enumerate(self.sinkhole_labels) if np.any(label == 1)]
        print(f"Total sinkhole patches: {len(self.sinkhole_patch_idxs)}")

        labelweights = np.zeros(2)
        num_point_all = []
        for sinkhole_sample, sinkhole_label in tqdm(zip(self.sinkhole_samples, self.sinkhole_labels), total=len(self.sinkhole_samples)):
            points, labels = sinkhole_sample[:, 0:3], sinkhole_label
            tmp, _ = np.histogram(labels, range(3))
            labelweights += tmp
            num_point_all.append(labels.size)

        self.labelweights = labelweights.astype(np.float32)
        self.labelweights = self.labelweights / np.sum(self.labelweights)
        print("非落水洞的占比：" + str(self.labelweights[0]) + "\n" + "落水洞的占比：" + str(self.labelweights[1]))

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(self.sinkhole_samples)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def sinkhole_hard_sampler(self):
        idx = np.random.choice(self.sinkhole_patch_idxs)
        points = self.sinkhole_samples[idx]
        labels = self.sinkhole_labels[idx]

        sinkhole_points = points[labels == 1]
        non_sinkhole_points = points[labels == 0]

        num_sinkhole_points = min(len(sinkhole_points), int(self.num_point * 0.3))
        num_non_sinkhole_points = self.num_point - num_sinkhole_points

        sinkhole_sampled = sinkhole_points[np.random.choice(len(sinkhole_points), num_sinkhole_points, replace=True)]
        non_sinkhole_sampled = non_sinkhole_points[np.random.choice(len(non_sinkhole_points), num_non_sinkhole_points, replace=True)]

        sampled_points = np.vstack((sinkhole_sampled, non_sinkhole_sampled))
        sampled_labels = np.hstack((np.ones(num_sinkhole_points), np.zeros(num_non_sinkhole_points)))

        shuffle_indices = np.random.permutation(self.num_point)
        return sampled_points[shuffle_indices], sampled_labels[shuffle_indices]

    def __getitem__(self, idx):
        if self.no_sample:
            sample_idx = self.room_idxs[idx]
            points = self.sinkhole_samples[sample_idx]
            labels = self.sinkhole_labels[sample_idx]
            return points.astype(np.float32), labels.astype(np.int64)

        if self.use_hard_sampler and np.random.rand() < 0.8:
            points, labels = self.sinkhole_hard_sampler()
        else:
            sample_idx = self.room_idxs[idx]
            points = self.sinkhole_samples[sample_idx]
            labels = self.sinkhole_labels[sample_idx]
            indices = np.random.choice(len(points), self.num_point, replace=True)
            points, labels = points[indices], labels[indices]

        if self.transform:
            transform_fn = random.choice([random_translation, random_rotation, random_scaling, add_noise])
            points = transform_fn(points)

        return points.astype(np.float32), labels.astype(np.int64)

    def __len__(self):
        return len(self.room_idxs)

