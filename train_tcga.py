import argparse
import copy
import datetime
import glob
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from collections import OrderedDict

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


import random
import numpy as np
import torch
import time

import torch

# 检查是否使用了 GPU
if torch.cuda.is_available():
    print(f"Current Device ID: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU is being used.")


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
    length = len(train_df)  # 样本数量

    # 记录训练开始时间
    start_time = time.time()

    for i in range(length):
        optimizer.zero_grad()

        # 获取当前样本的数据
        label, feats, single_label = get_bag_feats(train_df.iloc[i], args)  # label: [0,1] or [1,0]
        feats = dropout_patches(feats, args.dropout_patch)

        # 将数据转为 Tensor
        bag_label = Variable(Tensor([label]))
        bag_feats = Variable(Tensor([feats])).view(-1, args.feats_size)
        single_label = Variable(Tensor([single_label]))

        # 前向传播
        ins_prediction, bag_prediction, _, _, single_predictions, computed_instances_labels, mask_instances_labels = milnet(
            bag_feats, single_label
        )

        # 计算损失
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        instance_wise_loss = criterion(single_predictions, computed_instances_labels)
        averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()

        # 综合损失
        loss = 0.5 * bag_loss + 0.2 * max_loss + 0.3 * averaged_loss

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

    # 记录训练结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印样本数和训练时间
    log_and_print(f"Training: {length} samples processed in {elapsed_time:.2f} seconds")

    return total_loss / length


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
    sample_count = len(test_df)  # 记录测试样本数

    # 开始时间记录
    start_time = time.time()

    with torch.no_grad():
        for i in range(sample_count):
            label, feats, single_label = get_bag_feats(test_df.iloc[i], args)  # 获取特征
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats])).view(-1, args.feats_size)
            single_label = Variable(Tensor([single_label]))

            # 模型前向传播
            ins_prediction, bag_prediction, _, _, single_predictions, computed_instances_labels, mask_instances_labels = milnet(
                bag_feats, single_label
            )

            # 损失计算
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            instance_wise_loss = criterion(single_predictions, computed_instances_labels)
            averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()

            # 总损失
            loss = 0.4 * bag_loss + 0.3 * max_loss + 0.3 * averaged_loss
            total_loss += loss.item()

            # 记录标签和预测值
            test_labels.append(label)
            test_predictions.append(
                (0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()
            )

    # 结束时间记录
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 转换为 NumPy 数组
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # 计算多标签指标
    auc_value, _, thresholds_optimal, total_auc, Precision, Recall, Specificity, F1 = multi_label_roc(
        test_labels, test_predictions, args.num_classes, pos_label=1
    )

    # 单分类调整阈值
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        # 多分类调整阈值
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag

    # Bag 级别平均分数
    bag_score = sum(np.array_equal(test_labels[i], test_predictions[i]) for i in range(sample_count))
    avg_score = bag_score / sample_count

    # 打印样本数和时间
    log_and_print(f"Testing: {sample_count} samples processed in {elapsed_time:.2f} seconds")

    # 返回指标
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

import logging


# 设置日志功能
import logging


# 自定义函数，用于同时打印和记录日志
def log_and_print(message):
    print(message)
    logging.info(message)


def setup_logging(args):
    # 日志目录
    log_dir = '/data_disk/hj/wp/code/PAMIL/train_tcga'
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名为日期加上模型名称
    log_filename = os.path.join(log_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d")}_{args.model}_3pth.log')

    # 设置日志记录配置
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():

    parser = argparse.ArgumentParser(description='Train PAMIL on 20x patch features learned by SimCLR')
    # 添加 --seed 参数
    parser.add_argument('--seed', default=87, type=int, help='Random seed')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')  # default =2
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(1,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-single_20x', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='pamil', type=str, help='MIL model [pamil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--alpha', default=0.80, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--beta', default=0.20, type=float, help='Additional nonlinear operation [0]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

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
        bags_csv = "datasets/Came16-feat-single2/Came16-feat-single2.csv"
    else:
        bags_csv = os.path.join('/home/user/big_data/Wangping/pamil-wsi-master/datasets', args.dataset,
                                args.dataset + '.csv')

    bags_path = pd.read_csv(bags_csv, encoding="utf-8", error_bad_lines=False)
    train_path = bags_path.iloc[0:int(len(bags_path) * (1 - args.split)), :]
    test_path = bags_path.iloc[int(len(bags_path) * (1 - args.split)):, :]
    best_score = 0
    save_path = os.path.join('/data_disk/hj/wp/code/PAMIL/weights/0.9', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    for epoch in range(1, args.num_epochs):
        log_and_print(f"Epoch {epoch}/{args.num_epochs}")
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args)
        test_loss_bag, avg_score, aucs, Precision, Recall, Specificity, F1, thresholds_optimal, total_auc = test(
            test_path, milnet, criterion, optimizer, args)
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
