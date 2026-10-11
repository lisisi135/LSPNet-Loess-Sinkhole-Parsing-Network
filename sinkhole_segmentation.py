import argparse
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils.Dataloader import cityEnvDataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from data_utils.seg_method import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['non-sinkhole','sinkhole']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=r"/LSPNet/data/block", help='Log path [default:None]')# 替换为你的实际路径
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=16000, help='Point Number [default: 4096]')
    parser.add_argument('--lambda_nll', type=float, default=1.0, help='Weight for NLL loss [default: 1.0]')
    parser.add_argument('--lambda_tversky', type=float, default=1.0, help='Weight for Tversky loss [default: 1.0]')
    parser.add_argument('--lambda_bce', type=float, default=1.0, help='Weight for auxiliary BCE loss [default: 1.0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./'+args.log_dir+'/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    best_epoch = 0
    print("start loading training data ...")
    TRAIN_DATASET = cityEnvDataLoader(split='train', num_point=16000, data_root=r"/LSPNet/data/training_data", sample_rate=1.0,
                                transform=True, use_hard_sampler=True, no_sample=False)# 替换为你的实际路径
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    TEST_DATASET = cityEnvDataLoader(split='test', data_root=r"/LSPNet/data/testing_data", sample_rate=1.0,
                                transform=True, use_hard_sampler=False, no_sample=True)# 替换为你的实际路径
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)


    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss(
        lambda_nll=args.lambda_nll,
        lambda_tversky=args.lambda_tversky,
        lambda_bce=args.lambda_bce
    ).cuda()
    classifier.apply(inplace_relu)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.learning_rate / 1000, last_epoch=-1)



    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 40

    global_epoch = 0
    best_iou = 0
    best_acc = 0
    for epoch in range(0, args.epoch):
        savefile = []
        '''Train on chopped scenes'''
        log_string('*********************************************************************** Epoch %d (%d/%s) ***********************************************************************' % (global_epoch + 1, epoch + 1, args.epoch))
        current_lr = optimizer.param_groups[0]['lr']
        log_string('Learning rate:%f' % current_lr)
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        seg_pred_all, target_all = [],[]
        classifier = classifier.train()
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, aux_output = classifier(points)
            class_weights = torch.tensor([1.0, 16.0], device=points.device)

            loss = criterion(seg_pred, target, aux_output, class_weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.max(dim=2)[1]

            correct = (pred_choice == target).sum().item()
            total_correct += correct
            total_seen += BATCH_SIZE * NUM_POINT
            loss_sum += loss.item()
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if total_correct / float(total_seen) > best_acc:
            best_acc = total_correct / float(total_seen)
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        scheduler.step()
        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier

            log_string('---------------------------- EPOCH %03d EVALUATION ----------------------------' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                y_true = target.cpu().detach().numpy()
                for j in range(y_true.shape[0]):
                    seg_pred_all.append(seg_pred[j,:,:])
                    target_all.append(target[j,:])

                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)

                pred_result = torch.tensor(pred_val).cuda()

                save_data = torch.cat((points,pred_result.unsqueeze(1)),1)
                savefile.append(save_data.transpose(1,2).reshape(-1,4).cpu().numpy())
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = TEST_DATASET.labelweights.astype(np.float32) / np.sum(
                TEST_DATASET.labelweights.astype(np.float32))
            seg_pred_all = torch.cat(seg_pred_all, dim=0)
            target_all = torch.cat(target_all, dim=0)
            y_score_all = F.softmax(seg_pred_all, dim=-1)[ :, 1].detach().cpu().numpy()
            y_true_all = target_all.cpu().detach().numpy()
            fpr, tpr, _ = roc_curve(y_true_all.astype('int'), y_score_all)
            auc_score = auc(fpr, tpr)
            log_string('AUC: %f'%auc_score)
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('loss: %f' % (loss_sum / float(num_batches)))
            log_string('mIoU: %f' % (mIoU))
            log_string('OA: %f' % (total_correct / float(total_seen)))
            iou_per_class_str = ''
            for l in range(NUM_CLASSES):
                iou_per_class_str += '\nclass %s weight: %.3f, IoU: %.3f ' % (
                    seg_label_to_cat[l] , labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            log_string(iou_per_class_str)
            if mIoU >= best_iou:
                best_epoch = epoch
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
                visualize_dir = '/LSPNet/data/sunjiacha/block/visualize/'
                os.makedirs(visualize_dir, exist_ok=True)# 替换为你的实际路径
                best_vis_path = os.path.join(visualize_dir, 'best_prediction.txt')
                concatenated_data = np.concatenate(savefile, axis=0)
                np.savetxt(best_vis_path, concatenated_data)
            log_string('Best mIoU: %f' % best_iou + ' in epoch:%d'% (best_epoch+1)+'\n')
        global_epoch += 1
        concatenated_data = np.concatenate(savefile,axis=0)
    log_string('Best mIoU: %f' % best_iou + ' in epoch:%d'% (best_epoch+1))


if __name__ == '__main__':
    args = parse_args()
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    CUDA_LAUNCH_BLOCKING = 1
    main(args)
