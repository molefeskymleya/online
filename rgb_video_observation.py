from __future__ import division
import time
import os
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-model_file', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='1')
parser.add_argument('-dataset', type=str, default='charades')

args = parser.parse_args()

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms


import numpy as np
import json

import super_event
from apmeter import APMeter

#batch_size = 16
batch_size = 1
if args.dataset == 'multithumos':
    from multithumos_i3d_per_video import MultiThumos as Dataset
    from multithumos_i3d_per_video import mt_collate_fn as collate_fn
    train_split = 'data/multithumos.json'
    test_split = 'data/multithumos.json'
    rgb_root = '/ssd2/thumos/i3d_rgb'
    flow_root = '/ssd2/thumos/i3d_flow'
    classes = 65
elif args.dataset == 'charades':
    from charades_i3d_per_video import MultiThumos as Dataset
    from charades_i3d_per_video import mt_collate_fn as collate_fn
    train_split = '/root/liweiqi/my_test/55_my_charades.json'
    test_split = '/root/liweiqi/my_test/55_my_charades.json'
    rgb_root = '/root/liweiqi/my_test/55feats/rgb/'
    flow_root = '/root/liweiqi/super-events-cvpr18/pytorch-i3d/feat/flow/'
    classes = 157
elif args.dataset == 'ava':
    from ava_i3d_per_video import Ava as Dataset
    from ava_i3d_per_video import ava_collate_fn as collate_fn
    train_split = 'data/ava.json'
    test_split = train_split
    rgb_root = '/ssd2/ava/i3d_rgb'
    flow_root = '/ssd2/ava/i3d_flow'
    classes = 80
    # reduce batchsize as AVA videos are very long
    batch_size = 6
    


def sigmoid(x):
    return 1/(1+np.exp(-x))
#dataloaders, datasets = load_data(train_split, test_split, rgb_root)
def load_data(train_split, val_split, root):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:
        
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets

def get_new_feat(ori_feat,flattened_list):
    temp_feat = ori_feat[:,:,flattened_list[0]:flattened_list[0]+1]

    for i in flattened_list[1:]:
        if i != flattened_list[-1]:
            new_feat = ori_feat[:,:,i:(i+1)]
            updated_feat = torch.cat((temp_feat,new_feat),dim = 2)
            temp_feat = updated_feat
        elif i == flattened_list[-1]:
            new_feat = ori_feat[:,:,-1:]
            updated_feat = torch.cat((temp_feat,new_feat),dim = 2)
            temp_feat = updated_feat
    return temp_feat
        
        
def get_new_mask(ori_mask,flattened_list):
    temp_mask = ori_mask[:,flattened_list[0]:flattened_list[0]+1]
    for i in flattened_list[1:]:
        if i != flattened_list[-1]:
            new_mask = ori_mask[:,i:(i+1)]
            updated_mask = torch.cat((temp_mask,new_mask),dim = 1)
            temp_mask = updated_mask
        elif i == flattened_list[-1]:
            new_mask = ori_mask[:,-1:]
            updated_mask = torch.cat((temp_mask,new_mask),dim = 1)
            temp_mask = updated_mask
    return temp_mask
    
    
def get_new_label(ori_label,flattened_list):
    temp_label = ori_label[:,flattened_list[0]:flattened_list[0]+1]
    for i in flattened_list[1:]:
        if i != flattened_list[-1]:
            new_label = ori_label[:,i:(i+1)]
            updated_label = torch.cat((temp_label,new_label),dim = 1)
            temp_label = updated_label
        elif i == flattened_list[-1]:
            new_label = ori_label[:,-1:]
            updated_label = torch.cat((temp_label,new_label),dim = 1)
            temp_label = updated_label
    return temp_label

#updated_data = get_updated_data(data,flattened_list_32,num_feat)
def get_updated_data(data,flattened_list,num_feat):
    updated_data = []
    updated_feat = get_new_feat(data[0][:,:,0:num_feat],flattened_list)
    updated_mask = get_new_mask(data[1][:,0:num_feat],flattened_list)
    updated_label = get_new_label(data[2][:,0:num_feat],flattened_list)
    updated_data.append(updated_feat)
    updated_data.append(updated_mask)
    updated_data.append(updated_label)
    updated_data.append(data[3])
    return updated_data

def eval_model(model, dataloader, baseline=False):
    
    results_20 = {}
#    results_40 = {}
#    results_60 = {}
#    results_80 = {}
    
    for data in dataloader:
        num_feat = len(data[1].numpy()[0].tolist())
        other = data[3]
        num_feat_20 = int(num_feat*1)
        print('//////////////////////////')
        print(other[0])
        print(num_feat)
        if num_feat_20<32:
            print('frame numbers < 32')
            flattened_list = [x for x in range(0,num_feat_20)]
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20>32 and num_feat_20<64:
            print('frame numbers > 32 and < 64')
            left = [0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31]
            right = [x for x in range(32,num_feat_20,2)]
            flattened_list = left+right
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)

        elif num_feat_20==64:
            print('frame numbers = 64')
            flattened_list = [0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31, 32, 34, 36, 38, 40, 42, 44, 46, 49, 51, 53, 55, 57, 59, 61, 63]
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20>64 and num_feat_20<96:
            print('frame numbers > 64 and < 96')
            left = [0, 4, 9, 13, 18, 22, 27, 31, 32, 36, 41, 45, 50, 54, 59, 63]
            right = [x for x in range(64,num_feat_20,2)]
            flattened_list = left+right
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20 == 96:
            print('frame numbers = 96')
            flattened_list = [0, 4, 9, 13, 18, 22, 27, 31, 32, 36, 41, 45, 50, 54, 59, 63, 64, 66, 68, 70, 72, 74, 76, 78, 81, 83, 85, 87, 89, 91, 93, 95]
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20>96 and num_feat_20<128:
            print('frame numbers > 96 and <128')
            left = [0, 10, 21, 31, 32, 42, 53, 63, 64, 68, 73, 77, 82, 86, 91, 95]
            right = [x for x in range(96,num_feat_20,2)]
            flattened_list = left+right
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20 == 128:
            print('frame numbers =128')
            flattened_list = [0, 10, 21, 31, 32, 42, 53, 63, 64, 68, 73, 77, 82, 86, 91, 95, 96, 98, 100, 102, 105, 107, 109, 111, 113, 115, 117, 119, 122, 124, 126, 128]
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20>128 and num_feat_20<160:
            print('frame numbers > 128 and < 160')
            left = [0, 31, 32, 63, 64, 74, 85, 95, 96, 101, 105, 110, 114, 119, 123, 128]
            right = [x for x in range(129,num_feat_20,2)]
            flattened_list = left+right
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20==160:
            print('frame numbers = 160')
            flattened_list = [0, 31, 32, 63, 64, 74, 85, 95, 96, 101, 105, 110, 114, 119, 123, 128, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159]
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20>160 and num_feat_20<192:
            print('frame numbers > 160 and < 192')
            left = [0, 32, 64, 95, 96, 107, 117, 128, 129, 133, 138, 142, 146, 150, 155, 159]
            right = [x for x in range(160,num_feat_20,2)]
            
            flattened_list = left+right
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat_20>=192:
            print('frame numbers >=192')
            flattened_list = [0, 32, 64, 95, 96, 107, 117, 128, 129, 133, 138, 142, 146, 150, 155, 159, 160, 162, 164, 166, 168, 170, 172, 174, 177, 179, 181, 183, 185, 187, 189, 191]
            print(flattened_list)
            updated_data = get_updated_data(data,flattened_list,num_feat_20)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_20[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
    
    return results_20



def run_network(model, data, gpu, baseline=False):
    # get the inputs
    inputs, mask, labels, other = data
    
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))
    
    cls_wts = torch.FloatTensor([1.00]).cuda(gpu)

    # forward
    if not baseline:
        outputs = model([inputs, torch.sum(mask, 1)])
    else:
        outputs = model(inputs)
    outputs = outputs.squeeze(3).squeeze(3).permute(0,2,1) # remove spatial dims
    ##outputs = outputs.permute(0,2,1) # remove spatial dims
    probs = F.sigmoid(outputs) * mask.unsqueeze(2)
    
    # binary action-prediction loss
    loss = F.binary_cross_entropy_with_logits(outputs, labels, size_average=False)#, weight=cls_wts)

    
    loss = torch.sum(loss) / torch.sum(mask) # mean over valid entries
    
    # compute accuracy
    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs, loss, probs, corr/tot
            
  


if __name__ == '__main__':

    if args.mode == 'flow':
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)


    if args.train:
        pass

    else:
        print 'Evaluating...'
        rgb_model = torch.load(args.rgb_model_file)
        rgb_model.cuda()
        
        dataloaders, datasets = load_data('', test_split, rgb_root)
        rgb_results = eval_model(rgb_model, dataloaders['val'], baseline=False)

#        flow_model = torch.load(args.flow_model_file)
#        flow_model.cuda()
#        
#        dataloaders, datasets = load_data('', test_split, flow_root)
#        flow_results = eval_model(flow_model, dataloaders['val'], baseline=True)

        rapm = APMeter()
#        fapm = APMeter()
        tapm = APMeter()
        
        all_p_samples = 0
        all_n_samples = 0
        all_frames = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0


        for vid in rgb_results.keys():
            o,p,l,fps = rgb_results[vid]
            num_frame = sigmoid(o).shape[0]
            all_frames = all_frames+num_frame
            for i in range(num_frame):
                this_frame_label_num = 0
                this_frame_real_label_list = []
                this_frame_no_label_list = []
                
                this_frame_predict_array = sigmoid(o)[i]
                this_frame_predict_list = this_frame_predict_array.tolist()
                soretd_this_frame_predict_list = sorted(this_frame_predict_list,reverse = True)
                
                for j in range(157):
                    if l[i][j] == 1:
                        all_p_samples+=1
                        this_frame_label_num+=1
                        this_frame_real_label_list.append(j)
                         
                    if l[i][j] == 0:
                        all_n_samples+=1
                        this_frame_no_label_list.append(j)
                #get top-k scores, k means real label num
                this_frame_score_check_list = soretd_this_frame_predict_list[0:this_frame_label_num]
                #no real labels
                this_frame_score_no_check_list = soretd_this_frame_predict_list[this_frame_label_num:]
                this_frame_pred_list = []
                this_frame_no_pred_list = []
                
                for cid in this_frame_score_check_list:
                    for k in range(len(this_frame_predict_list)):
                        if cid == this_frame_predict_list[k]:
                            this_frame_pred_list.append(k)
                            
                #for compute tn
                for cid in this_frame_score_no_check_list:
                    for k in range(len(this_frame_predict_list)):
                        if cid == this_frame_predict_list[k]:
                            this_frame_no_pred_list.append(k)
                            
                #tp
                for i in this_frame_real_label_list:
                    if i in this_frame_pred_list:
                        tp+=1
                
#                #fp
#                for i in this_frame_no_label_list:
#                    if i in this_frame_pred_list:
#                        fp+=1
                #fp
                for i in this_frame_pred_list:
                    if i not in this_frame_real_label_list:
                        fp+=1
                        
                        
                        
                        
                #fn
                for i in this_frame_real_label_list:
                    if i not in this_frame_pred_list:
                        fn+=1
                        
                #for compute tn
                for i in this_frame_no_label_list:
                    if i not in this_frame_pred_list:
                        tn+=1
            rapm.add(sigmoid(o), l)
#            fapm.add(sigmoid(flow_results[vid][0]), l)
#            if vid in flow_results:
#                o2,p2,l2,fps = flow_results[vid]
#                o = (o[:o2.shape[0]]*.5+o2*.5)
#                p = (p[:p2.shape[0]]*.5+p2*.5)
#            tapm.add(sigmoid(o), l)
        print 'rgb MAP:', rapm.value().mean()
        print(all_frames)
        print(all_p_samples)
        print(all_n_samples)
        print(tp)
        print(fp)
        print(tn)
        print(fn)
        print('accuracy')
        print((tp+tn)/(tp+fp+tn+fn))
        print('precision')
        print(tp/(tp+fp))
        print('recall')
        print(tp/(tp+fn))
#        print 'flow MAP:', fapm.value().mean()
#        print 'two-stream MAP:', tapm.value().mean()
