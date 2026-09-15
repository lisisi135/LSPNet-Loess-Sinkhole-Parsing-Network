import torch.nn as nn
import torch.nn.functional as F
import torch
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, knn_point

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.num_classes = num_classes
        self.sa1 = PointNetSetAbstraction(npoint=4000, radius=0.1, nsample=32, in_channel=3 + 3, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1000, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=250, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=60,  radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, kernel_size=1)
        self.aux_head = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, xyz):
        l0_points = xyz[:, :3, :]
        l0_min = torch.min(l0_points, dim=-1, keepdim=True)[0]
        l0_points = (l0_points - l0_min) / 10.0
        l0_xyz = l0_points

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        aux_out = self.aux_head(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x, (aux_out, l2_xyz, l0_xyz)

class get_loss(nn.Module):
    def __init__(self, lambda_nll=1.0, lambda_tversky=1.0, lambda_bce=1.0):
        super(get_loss, self).__init__()
        self.alpha = 0.7
        self.beta = 0.3
        self.lambda_nll = lambda_nll
        self.lambda_tversky = lambda_tversky
        self.lambda_bce = lambda_bce

    def forward(self, pred, target, trans_feat, weight=None):
        """
        Compute weighted loss fusion:
        lambda_nll * NLL loss + lambda_tversky * Tversky loss + lambda_bce * BCE loss.
        Class weights are applied to NLL and auxiliary BCE losses.
        """
        if isinstance(trans_feat, tuple):
            aux_pred, l2_xyz, l0_xyz = trans_feat
        else:
            return F.nll_loss(pred, target, weight=weight)

        if target.dim() > 1:
            B, N = target.size(0), target.size(1)
            pred_flat = pred.reshape(-1, pred.shape[-1])
            target_flat = target.reshape(-1)
        else:
            pred_flat = pred
            target_flat = target
            B = aux_pred.size(0)
            N = pred_flat.size(0) // B
            target = target_flat.view(B, N)

        main_loss = F.nll_loss(pred_flat, target_flat, weight=weight)

        if pred.dim() == 2:
            pred_probs = pred_flat.exp().view(B, N, -1)
        else:
            pred_probs = pred.exp()
        sink_idx = 1
        pred_sink_prob = pred_probs[..., sink_idx] if pred_probs.shape[-1] > sink_idx else pred_probs.squeeze(-1)
        target_mask = (target == sink_idx).float()

        TP = (pred_sink_prob * target_mask).sum()
        FN = (target_mask * (1 - pred_sink_prob)).sum()
        FP = ((1 - target_mask) * pred_sink_prob).sum()
        eps = 1e-6
        if target_mask.sum() == 0:

            if pred_sink_prob.sum() < eps:
                tversky_loss = torch.tensor(0.0, device=pred.device)
            else:
                tversky_index = TP / (TP + self.alpha * FN + self.beta * FP + eps)
                tversky_loss = 1.0 - tversky_index
        else:
            tversky_index = TP / (TP + self.alpha * FN + self.beta * FP + eps)
            tversky_loss = 1.0 - tversky_index

        idx = knn_point(1, l0_xyz.permute(0, 2, 1).contiguous(), l2_xyz.permute(0, 2, 1).contiguous())
        idx = idx.squeeze(-1)
        aux_target = target.gather(1, idx)
        aux_target_mask = aux_target.float()
        aux_logits = aux_pred.squeeze(1)
        bce_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_target_mask, reduction='none')
        if weight is not None and weight.numel() == 2:
            weight_mask = aux_target_mask * weight[1] + (1 - aux_target_mask) * weight[0]
            bce_loss = bce_loss * weight_mask
        aux_loss = bce_loss.mean()

        total_loss = (
            self.lambda_nll * main_loss +
            self.lambda_tversky * tversky_loss +
            self.lambda_bce * aux_loss
        )
        return total_loss


if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(1, 9, 16000)
    print(model(xyz))
