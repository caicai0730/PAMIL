import argparse
import copy
import datetime
import glob
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from collections import OrderedDict
import random
import numpy as np
import torch
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as VF
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

def set_seed(seed):
    if seed is None:
        # 生成一个基于时间的随机种子
        seed = int(time.time()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU

    # 确保使用固定算法进行计算（不使用非确定性算法）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_bag_feats(csv_file_df, args): # csv_file_df 一个包含csvpath的列表 ； 现在传进来的是numpy
    # 1 2 3 4 5 csv文件  -> csv文件
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = '/home/user/big_data/Wangping/pamil-wsi-master/datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]  #应该是一个csv文件的路径
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)  # csv中label为1 == label[0,1] ;label为0 == label[1,0]
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:   #default num_classes = 2
        # i = csv_file_df.iloc[1]
        # l = (len(label)-1)
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1

    single_label = label

    return label, feats,single_label # label:[0,1] or [1,0]

# def train(train_df, milnet, criterion, optimizer, args): #train_path
#     milnet.train()
#     csvs = shuffle(train_df).reset_index(drop=True) #right
#     total_loss = 0
#     bc = 0
#     Tensor = torch.cuda.FloatTensor
#     length = len(train_df)
#     for i in range(len(train_df)):
#         optimizer.zero_grad()
#         train_df_cur_iloc = train_df.iloc[i]
#
#         # label在这里  要么是0 要么是1
#         # csv中label为1 == label[0,1] ;label为0 == label[1,0]
#         # 如果bag_label == [0,1] single_label = 1
#         # 如果bag_label == [1,0] single_label = 0
#         label, feats,single_label = get_bag_feats(train_df.iloc[i], args) # label:[0,1] or [1,0]
#         feats = dropout_patches(feats, args.dropout_patch)
#
#         bag_label = Variable(Tensor([label]))
#         bag_feats = Variable(Tensor([feats])) #[ _ ,512]
#         bag_feats = bag_feats.view(-1, args.feats_size)
#
#         # milnet里训练一个分类器 ，返回另一篇论文 seg的instance_predictions
#         # 损失这里加一个 seg的损失 这个损失该是多大呢 0.4+0.4+0.2？ 或许可以
#         single_label = Variable(Tensor([single_label]))
#
#         ins_prediction, bag_prediction, _, _,single_predictions,computed_instances_labels,mask_instances_labels = milnet(bag_feats,single_label) # 这里返回一个single_prediction
#         max_prediction, _ = torch.max(ins_prediction, 0)
#         bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
#         max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
#
#         # 分类器损失
#         instance_wise_loss = criterion(single_predictions, computed_instances_labels)
#         averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()
#
#
#         # loss = 0.5*bag_loss + 0.5*max_loss
#         loss = 0.5*bag_loss + 0.2*max_loss + 0.3*averaged_loss
#
#         loss.backward()
#         optimizer.step()
#         total_loss = total_loss + loss.item()
#         #sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
#     return total_loss / len(train_df)
def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)  # shuffle training data
    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    length = len(train_df)  # 样本数
    bc = 0

    # 记录训练时间
    start_time = time.time()

    for i in range(len(train_df)):
        optimizer.zero_grad()
        train_df_cur_iloc = train_df.iloc[i]
        label, feats, single_label = get_bag_feats(train_df.iloc[i], args)
        feats = dropout_patches(feats, args.dropout_patch)

        bag_label = Variable(Tensor([label]))
        bag_feats = Variable(Tensor([feats])).view(-1, args.feats_size)

        single_label = Variable(Tensor([single_label]))
        ins_prediction, bag_prediction, _, _, single_predictions, computed_instances_labels, mask_instances_labels = milnet(
            bag_feats, single_label
        )
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        instance_wise_loss = criterion(single_predictions, computed_instances_labels)
        averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()
        loss = 0.5 * bag_loss + 0.2 * max_loss + 0.3 * averaged_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    end_time = time.time()
    training_time = end_time - start_time
    log_and_print(f"Training: {length} samples processed in {training_time:.2f} seconds")

    return total_loss / len(train_df)


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

# def test(test_df, milnet, criterion, optimizer, args):
#     milnet.eval()
#     csvs = shuffle(test_df).reset_index(drop=True)
#     total_loss = 0
#     test_labels = []
#     test_predictions = []
#     Tensor = torch.cuda.FloatTensor
#     with torch.no_grad():
#         for i in range(len(test_df)):
#             label, feats,single_label = get_bag_feats(test_df.iloc[i], args) #在这里生成新特征
#             bag_label = Variable(Tensor([label]))
#             bag_feats = Variable(Tensor([feats]))
#             bag_feats = bag_feats.view(-1, args.feats_size)
#
#
#             # milnet里训练一个分类器 ，返回另一篇论文 seg的instance_predictions
#             # 损失这里加一个 seg的损失 这个损失该是多大呢 0.4+0.4+0.2？ 或许可以
#             single_label = Variable(Tensor([single_label]))
#
#             ins_prediction, bag_prediction, _, _,single_predictions,computed_instances_labels,mask_instances_labels = milnet(bag_feats,single_label) # 这里返回一个single_prediction
#             max_prediction, _ = torch.max(ins_prediction, 0)
#             bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
#             max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
#
#             # 分类器损失
#             instance_wise_loss = criterion(single_predictions, computed_instances_labels)
#             averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()
#
#
#             # loss = 0.5*bag_loss + 0.5*max_loss
#             loss = 0.4*bag_loss + 0.3*max_loss + 0.3*averaged_loss
#
#
#
#             # ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
#             # max_prediction, _ = torch.max(ins_prediction, 0)
#             # bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
#             # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
#             # loss = 0.5*bag_loss + 0.5*max_loss
#
#             total_loss = total_loss + loss.item()
#             #sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
#             test_labels.extend([label])
#             test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
# #             test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
#     test_labels = np.array(test_labels)
#     test_predictions = np.array(test_predictions)
#     auc_value, _, thresholds_optimal, total_auc, Precision, Recall, Specificity, F1 = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
#     if args.num_classes==1:
#         class_prediction_bag = copy.deepcopy(test_predictions)
#         class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
#         class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
#         test_predictions = class_prediction_bag
#         test_labels = np.squeeze(test_labels)
#     else:
#         for i in range(args.num_classes):
#             class_prediction_bag = copy.deepcopy(test_predictions[:, i])
#             class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
#             class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
#             test_predictions[:, i] = class_prediction_bag
#     bag_score = 0
#     for i in range(0, len(test_df)):
#         bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
#     avg_score = bag_score / len(test_df)
#
#     # 精确率、召回率、F1分数
#     # TP = all_predict_true = all_lable_true = 0
#     # for i in range(0, len(test_df)):
#     #     if test_predictions[i][1] == True:
#     #         TP = np.array_equal(test_labels[i], test_predictions[i]) + TP
#     #         all_predict_true = all_predict_true + 1
#     #     if test_labels[i][1] == True:
#     #         all_lable_true = all_lable_true + 1
#     # Precision = TP / all_predict_true
#     # Recall = TP / all_lable_true
#     # F1 = 2 * Precision * Recall / (Precision + Recall)
#
#     return total_loss / len(test_df), avg_score, auc_value, Precision, Recall, Specificity, F1, thresholds_optimal, total_auc
def test(test_df, milnet, criterion, optimizer, args):
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor

    # 记录测试时间
    start_time = time.time()
    sample_count = len(test_df)  # 统计测试样本数量

    with torch.no_grad():
        for i in range(sample_count):
            # 读取样本数据
            label, feats, single_label = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats])).view(-1, args.feats_size)
            single_label = Variable(Tensor([single_label]))

            # 进行模型推理
            ins_prediction, bag_prediction, _, _, single_predictions, computed_instances_labels, mask_instances_labels = milnet(
                bag_feats, single_label
            )

            # 计算损失
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            instance_wise_loss = criterion(single_predictions, computed_instances_labels)
            averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()
            loss = 0.4 * bag_loss + 0.3 * max_loss + 0.3 * averaged_loss

            # 累计损失
            total_loss += loss.item()

            # 记录预测结果和标签
            test_labels.append(label)
            test_predictions.append(
                (0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()
            )

    # 测试时间统计
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 转换预测结果为 NumPy 数组
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # 计算性能指标
    auc_value, _, thresholds_optimal, total_auc, Precision, Recall, Specificity, F1 = multi_label_roc(
        test_labels, test_predictions, args.num_classes, pos_label=1
    )

    # 针对单分类问题的阈值调整
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        # 针对多分类问题的阈值调整
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag

    # 计算 bag-level 的平均分数
    bag_score = 0
    for i in range(sample_count):
        bag_score += np.array_equal(test_labels[i], test_predictions[i])
    avg_score = bag_score / sample_count

    # 输出测试时间和样本统计信息
    log_and_print(f"Testing: {sample_count} samples processed in {elapsed_time:.2f} seconds")

    # 返回所有指标
    return total_loss / sample_count, avg_score, auc_value, Precision, Recall, Specificity, F1, thresholds_optimal, total_auc


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    # fprs = []
    # tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    precisionss, recalls, specificitys, F1s = 0, 0, 0, 0
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)  # FPR假正率，TPR真正率
        precision, recall, specificity, F1= eval_metric(prediction, label)
        precisionss = precisionss + precision.item()
        recalls = recalls + recall.item()
        specificitys = specificitys + specificity.item()
        F1s = F1s + F1
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    # total_auc = roc_auc_score(labels, predictions,average="micro")
    total_auc2 = roc_auc_score(labels, predictions, average="macro")
    return aucs, thresholds, thresholds_optimal, total_auc2, (precisionss / num_classes), (recalls / num_classes),\
           (specificitys / num_classes), (F1s / num_classes)  # ,total_auc

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)#    c_auc = roc_auc_score(label, prediction)#
    return c_auc, threshold_optimal

def eval_metric(oprob, label):
    _, threshold = roc_threshold(label, oprob)
    prob = oprob > threshold
    label = label > threshold

    TP = torch.tensor((prob & label).sum(0), dtype=torch.float32)
    TN = torch.tensor(((~prob) & (~label)).sum(0), dtype=torch.float32)
    FP = torch.tensor((prob & (~label)).sum(0), dtype=torch.float32)
    FN = torch.tensor(((~prob) & label).sum(0), dtype=torch.float32)

    # accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    Precision = torch.mean(TP / (TP + FP + 1e-12))
    Recall = torch.mean(TP / (TP + FN + 1e-12))
    Specificity = torch.mean(TN / (TN + FP + 1e-12))
    F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-12)

    return Precision, Recall, Specificity, F1

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

# 设置日志功能
import logging


# 自定义函数，用于同时打印和记录日志
def log_and_print(message):
    print(message)
    logging.info(message)

def setup_logging(args):
    # 日志目录
    log_dir = '/data_disk/hj/wp/code/PAMIL/train_C16'
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名为日期加上模型名称
    log_filename = os.path.join(log_dir, f'10xn5t5zhanbi{datetime.datetime.now().strftime("%Y-%m-%d")}_{args.model}_alpha:{args.alpha}beta:{args.beta}_2pth.log')

    # 设置日志记录配置
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description='Train pamil on 20x patch features learned by SimCLR')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]') #default =2
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset',default='Came16-feat-single2', type=str, help='Dataset folder name')
    #default TCGA-lung-default Camelyon16-multiscale TCGA-lungfeat-single Came16-feat-single Came16-feat-single2
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='pamil', type=str, help='MIL model [pamil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--alpha', default=0.15, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--beta', default=0.60, type=float, help='Additional nonlinear operation [0]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

#
#     if args.model == 'abmil':
#         import abmil as mil
#
#     elif args.model == 'pamil':
#         import pamil as mil
#     i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes)  # 512->2
#     b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
#
#     # 这里是分一类
#     s_classifier = mil.SingleClassifier(feature_size=args.feats_size,out_class=1,alpha=args.alpha,beta=args.beta)
#
#     #milnet = mil.MILNet(i_classifier, b_classifier).cuda()
#     milnet = mil.MILNet(i_classifier, b_classifier,s_classifier).cuda()
#
#     ###这里可能会出问题
#     if args.model == 'pamil':
#         state_dict_weights = torch.load('/data_disk/hj/wp/copy/init.pth')#init.pth --> 加载权重
#         try:
#             milnet.load_state_dict(state_dict_weights, strict=False)
#         except:
#             del state_dict_weights['b_classifier.v.1.weight']
#             del state_dict_weights['b_classifier.v.1.bias']
#             milnet.load_state_dict(state_dict_weights, strict=False)
#     criterion = nn.BCEWithLogitsLoss()
#
#
#     optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
#
#     if args.dataset == 'TCGA-lung-default':
#         bags_csv = '/home/user/big_data/Wangping/pamil-wsi-master/datasets/tcga-dataset/TCGA.csv'
#     # elif args.dataset == "Camelyon16":
#     #
#     #     bags_csv = "F:/dataset_csv/camelyon16/fold0_1.csv"
#     elif args.dataset == "TCGA-lungfeat-single":
#
#         bags_csv = "datasets/TCGA-lungfeat-single/TCGA-lungfeat-single.csv"
#     elif args.dataset == "TCGA-lung-single_20x":
#
#         bags_csv = "/data_disk/hj/wp/dataset/TCGA-lung-single_20x/TCGA-lungfeat-single2.csv"
#     elif args.dataset == "Came16-feat-single":
#
#         bags_csv = "/data_disk/hj/wp/dataset/Camelyon16-0-10x/Came16-feat-single2.csv"
#     elif args.dataset == "Came16-feat-single2":
#
#         bags_csv = "/data_disk/hj/wp/dataset/Came16-feat-single3/single/Came16-feat-single3.csv"
#
#     else:
#         # bags_csv = os.path.join('/home/user/big_data/Wangping/pamil-wsi-master/datasets', args.dataset, args.dataset+'.csv')
#         bags_csv = "datasets/Camelyon16-multiscale/Camelyon16-multiscale.csv"
#
#     # bags_path = pd.concat(map(pd.read_csv, bags_csv))
#     bags_path = pd.read_csv(bags_csv)
#     train_path = bags_path.iloc[0:int(len(bags_path)*(1-args.split)), :]
#     test_path = bags_path.iloc[int(len(bags_path)*(1-args.split)):, :]
#     best_score = 0
#     save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
#     os.makedirs(save_path, exist_ok=True)
#     run = len(glob.glob(os.path.join(save_path, '*.pth')))
#     for epoch in range(1, args.num_epochs):
#         train_path = shuffle(train_path).reset_index(drop=True)
#         test_path = shuffle(test_path).reset_index(drop=True)
#         train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
#         test_loss_bag, avg_score, aucs, Precision, Recall, F1, thresholds_optimal, total_auc = test(test_path, milnet, criterion, optimizer, args)
#         if args.dataset=='TCGA-lung':
#             print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, marco: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' %
#                   (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, Precision, Recall, F1, total_auc, aucs[0], aucs[1]))
#         else:
#             print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC: ' %
#                   (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, Precision, Recall, F1) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))
#         scheduler.step()
#         current_score = (sum(aucs) + avg_score)/2
#         if current_score >= best_score:
#             best_score = current_score
#             save_name = os.path.join(save_path, str(run+1)+'.pth')
#             torch.save(milnet.state_dict(), save_name)
#             if args.dataset=='TCGA-lung':
#                 print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
#             else:
#                 print('Best model saved at: ' + save_name)
#                 print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
#
#
# if __name__ == '__main__':
#     main()

    # 调用日志配置
    setup_logging(args)
    set_seed(args.seed)
    if args.model == 'abmil':
        import abmil as mil
    elif args.model == 'pamil':
        import dsmil_tcga as mil

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes)  # 512->2
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                   dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()

    s_classifier = mil.SingleClassifier(feature_size=args.feats_size, out_class=1, alpha=args.alpha, beta=args.beta)
    milnet = mil.MILNet(i_classifier, b_classifier, s_classifier).cuda()

    if args.model == 'pamil':
        state_dict_weights = torch.load('/data_disk/hj/wp/code/PAMIL/init.pth')
        try:
            milnet.load_state_dict(state_dict_weights, strict=False)
        except:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            milnet.load_state_dict(state_dict_weights, strict=False)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    if args.dataset == 'TCGA-lung-default':
        bags_csv = '/home/user/big_data/Wangping/pamil-wsi-master/datasets/tcga-dataset/TCGA.csv'
    elif args.dataset == "TCGA-lungfeat-single":
        bags_csv = "datasets/TCGA-lungfeat-single/TCGA-lungfeat-single3.csv"
    elif args.dataset == "TCGA-lung-single_20x":
        bags_csv = "/data_disk/hj/wp/dataset/TCGA-lung/TCGA-lung.csv"
    elif args.dataset == "Came16-feat-single":
        bags_csv = "datasets/Came16-feat-single/Came16-feat-single.csv"
    elif args.dataset == "Came16-feat-single2":
        bags_csv = "/data_disk/hj/wp/dataset/Camelyon16/pyramid/n5t5_combined.csv"
    else:
        bags_csv = os.path.join('/home/user/big_data/Wangping/pamil-wsi-master/datasets', args.dataset,
                                args.dataset + '.csv')

    bags_path = pd.read_csv(bags_csv, encoding="utf-8", error_bad_lines=False)
    train_path = bags_path.iloc[0:int(len(bags_path) * (1 - args.split)), :]
    test_path = bags_path.iloc[int(len(bags_path) * (1 - args.split)):, :]
    best_score = 0


    ##————————————————————————————————模型保存————————————————————————————————————
    save_path = os.path.join('/data_disk/hj/wp/code/PAMIL/weights/10x_0.15_0.6_n5t5zhanbi', datetime.date.today().strftime("%m%d%Y"))
    ##———————————————————————————————————————————————————————————————————————————


    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    # for epoch in range(1, args.num_epochs):
    #     train_path = shuffle(train_path).reset_index(drop=True)
    #     test_path = shuffle(test_path).reset_index(drop=True)
    #     train_loss_bag = train(train_path, milnet, criterion, optimizer, args)
    #     test_loss_bag, avg_score, aucs, Precision, Recall, Specificity, F1, thresholds_optimal, total_auc = test(
    #         test_path, milnet, criterion, optimizer, args)

    for epoch in range(1, args.num_epochs):
        log_and_print(f"Epoch {epoch}/{args.num_epochs}")

        # 训练阶段
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args)

        # 测试阶段
        test_loss_bag, avg_score, aucs, Precision, Recall, Specificity, F1, thresholds_optimal, total_auc = test(
            test_path, milnet, criterion, optimizer, args
        )

        log_and_print(f"Epoch {epoch}/{args.num_epochs} completed")

        if args.dataset == 'TCGA-lung':
            log_and_print(
                f'Epoch [{epoch}/{args.num_epochs}] train loss: {train_loss_bag:.4f} test loss: {test_loss_bag:.4f}, average score: {avg_score:.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, Specificity: {Specificity:.4f}, F1: {F1:.4f}, marco_auc: {total_auc:.4f}, auc_LUAD: {aucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}')
        else:
            log_and_print(
                f'Epoch [{epoch}/{args.num_epochs}] train loss: {train_loss_bag:.4f} test loss: {test_loss_bag:.4f}, average score: {avg_score:.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, Specificity: {Specificity:.4f}, F1: {F1:.4f}, AUC: {"|".join(f"class-{k}>>{v:.4f}" for k, v in enumerate(aucs))}')

        scheduler.step()

        current_score = (sum(aucs) + avg_score) / 2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, f'{run + 1}.pth')
            torch.save(milnet.state_dict(), save_name)
            if args.dataset == 'TCGA-lung':
                log_and_print(
                    f'Best model saved at: {save_name} Best thresholds: LUAD {thresholds_optimal[0]:.4f}, LUSC {thresholds_optimal[1]:.4f}')
            else:
                log_and_print(f'Best model saved at: {save_name}')
                log_and_print('Best thresholds ===>>> ' + '|'.join(
                    f'class-{k}>>{v:.4f}' for k, v in enumerate(thresholds_optimal)))


if __name__ == '__main__':
    main()
