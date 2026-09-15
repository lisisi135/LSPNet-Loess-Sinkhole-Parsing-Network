
import torch
from tqdm import tqdm
from models import pointnet2_sem_seg
from data_utils.seg_method import *
filepath = r"/test.txt"# 替换为你的实际路径
savename = r"/sunjiacha_pre.txt"# 替换为你的实际路径
whole_place = np.loadtxt(filepath)[:,0:3]
point_idx = np.argsort(whole_place[:,1])
point_idx_reverse = np.argsort(whole_place[:,1])[::-1]
blocks = []
if(determineCutCount(whole_place) == 0):
     seg80000(blocks,[],whole_place,filepath)
else:
     seg16000(blocks, [], whole_place, filepath)
testDataLoader = torch.utils.data.DataLoader(blocks, batch_size=2, shuffle=False, num_workers=0,pin_memory=True, drop_last=True)
checkpoint = torch.load(r"/LSPNet/data/block/checkpoints/best_model.pth")# 替换为你的实际路径
classifier = pointnet2_sem_seg.get_model(2).cuda()
classifier.load_state_dict(checkpoint['model_state_dict'])
savefile = []
for i, points in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
     points = points.data.numpy()
     points = torch.Tensor(points)
     points = points.transpose(2, 1)
     seg_pred, trans_feat = classifier(points.cuda())
     pred_val = seg_pred.contiguous().cpu().data.numpy()
     pred_val = np.argmax(pred_val, 2)
     pred_result = torch.tensor(pred_val).cuda()
     save_data = torch.cat((points.cuda(),pred_result.unsqueeze(1)),1)
     savefile.append(save_data.transpose(1,2).reshape(-1,4).cpu().numpy())
concat = np.concatenate(savefile,axis=0)
np.savetxt(savename,concat)



