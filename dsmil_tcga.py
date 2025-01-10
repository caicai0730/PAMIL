import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchextractor as tx


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x



# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution without padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

# class Conv2d_BN(nn.Module):
#     """Convolution with BN module."""
#     def __init__(
#             self,
#             in_ch,
#             out_ch,
#             kernel_size=1,
#             stride=1,
#             pad=0,
#             dilation=1,
#             groups=1
#     ):
#         super().__init__()
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.conv = torch.nn.Conv2d(in_ch,
#                                     out_ch,
#                                     kernel_size,
#                                     stride,
#                                     pad,
#                                     dilation,
#                                     groups,
#                                     bias=False
#         )
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.act_layer = nn.ReLU()

#     def forward(self, x):
#         # ain = self.in_ch
#         # aout = self.out_ch

#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act_layer(x)
        
#         return x


# class IClassifier_h(nn.Module):
#     def __init__(self, feature_extractor, feature_size, output_class):
#         super(IClassifier_h, self).__init__()
        
#         self.feature_extractor = feature_extractor   

#         self.feature_ex = tx.Extractor(self.feature_extractor, ["layer1", "layer2", "layer3", "layer4"])  

#         self.layer2_outconv = conv1x1(128,256)
#         self.layer3_outconv = conv1x1(256,512)

        
        
        
#         self.fc = nn.Linear(feature_size, output_class)
        
        
#     def forward(self, x):
#         device = x.device
#         # feats = self.feature_extractor(x) # N x K

#         feature_output,features = self.feature_ex(x) #  features = layer1:out1
#         feature_outs = {name: f for name, f in features.items()}
#         in_ch = 0
#         out_ch = feature_outs['layer4'].shape[1]

        
#         layer4_2x = F.interpolate(feature_outs['layer4'],scale_factor=2.,mode='bilinear', align_corners=True)
#         layer3 = self.layer3_outconv(feature_outs['layer3'])


#         layer3_2x = F.interpolate(feature_outs['layer3'],scale_factor=2.,mode='bilinear', align_corners=True)
        

        





#         c = self.fc(feats.view(feats.shape[0], -1)) # N x C
#         return feats.view(feats.shape[0], -1), c

# class IClassifier_h(nn.Module):
#     def __init__(self, feature_extractor, feature_size, output_class):
#         super(IClassifier_l, self).__init__()
        
#         self.feature_extractor = feature_extractor      
#         self.fc = nn.Linear(feature_size, output_class)
        
        
#     def forward(self, x):
#         device = x.device
#         feats = self.feature_extractor(x) # N x K
#         c = self.fc(feats.view(feats.shape[0], -1)) # N x C
#         return feats.view(feats.shape[0], -1), c


class SingleClassifier(nn.Module):
    def __init__(self,feature_size,out_class,alpha,beta): #分类的输出头 single_class
        super(SingleClassifier, self).__init__()
             
        self.single_classifier = nn.Linear(feature_size, out_class)
        #新参数
        self.alpha = alpha
        self.beta = beta
        #,alpha,beta,single_label

    def forward(self,x,single_label):
        device = x.device
        single_predictions = self.single_classifier(x) # output single_classes  

        # 计算instance的proxy-label ，在给定bag label，alpha，beta，和预测结果上
        n_instances = single_predictions.size(0)
        computed_instances_labels = torch.zeros(single_predictions.shape,device=device).float()
        mask_instances_labels = torch.zeros(single_predictions.shape,device=device).float()
        if single_label == [1.,0.]:
            # computed_instances_labels[:] = 0.
            # mask_instances_labels[:] = 1.
            _, topk_idx = torch.topk(single_predictions, k=int(self.alpha*n_instances), dim=0)
            computed_instances_labels[topk_idx] = 0.
            mask_instances_labels[topk_idx] = 1.
        else:
            _, topk_idx = torch.topk(single_predictions, k=int(self.alpha*n_instances), dim=0)
            computed_instances_labels[topk_idx] = 1.
            mask_instances_labels[topk_idx] = 1.
            # if self.beta > 0.:
            #     _, bottomk_idx = torch.topk(single_predictions, k=int(self.beta*n_instances), largest=False, dim=0)
            #     computed_instances_labels[bottomk_idx] = 0.
            #     mask_instances_labels[bottomk_idx] = 1.

        computed_instances_labels = computed_instances_labels.detach()
        mask_instances_labels = mask_instances_labels.detach()

        return single_predictions,computed_instances_labels,mask_instances_labels #single_predictions-shape: [N,1]

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x): # x:[128,3,256,256]
        # 推理阶段 先放弃 多尺度特征


        feats = self.feature_extractor(x) # N x K   # 128 x 512
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C  # 128 x 2
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module): 
    #input:512 output:2
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  


        
    def forward(self, feats, c,single_predictions): # N x K, N x C K:512 C:2  N x 1
        device = feats.device
        instance_predictions = single_predictions
        n_instance = instance_predictions.size(0)
        
        
        _,lastk_idx = torch.topk(instance_predictions,k=int(0.20*n_instance),largest=False,dim = 0) #0.6
        instance_predictions[lastk_idx] = 0.
        

        instance_score = F.softmax(instance_predictions,dim=0)  # [N,1]
        # instance_score 乘以 feats
        feats = torch.mul(feats,instance_score) + feats

        # feats = torch.mul(feats,instance_score)

        feats = self.lin(feats)

        V = self.v(feats) # N x V, unsorted #512
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted #128


        


        # handle multiple classes without for loop
        # torch.sort() 按指定的维度进行排序，descending=True递减； 
        # 返回值（Tensor，indices）-> (排序后的tensor, 原始tensor中元素索引组成的tensor)
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        # i = m_indices[0,:]
        # index_select(): 在patch维度上，选择
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        # a = torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))

        #####
            # 要不这里也选一定比例的 实例 进行归一化？ 
        #####
        
        # _,index = torch.topk(A,k=int(0.4*n_instance),largest=False,dim=0) #0.4
        # A[index] = 0.

        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        
        # [N,C] x [N,V]
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1) # 1 x C
        return C, A, B
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier,s_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
        self.s_classifier = s_classifier


    def forward(self, x,single_label):
        # feats 就是 输入的特征x
        feats, classes = self.i_classifier(x) # feats:[:512] classes:[:2]

        single_predictions,computed_instances_labels,mask_instances_labels = self.s_classifier(x,single_label)

        prediction_bag, A, B = self.b_classifier(feats, classes,
                                                single_predictions
                                                )
        
        return classes, prediction_bag, A, B,single_predictions,computed_instances_labels,mask_instances_labels
        