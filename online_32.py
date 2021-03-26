from __future__ import division
import time
import itertools
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

import time
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
    train_split = '/root/liweiqi/super-events-cvpr18/data/test_charades.json'
    test_split = '/root/liweiqi/super-events-cvpr18/data/test_charades.json'
    rgb_root = '/root/liweiqi/super-events-cvpr18/pytorch-i3d/feat/rgb/'
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


#updated_feat = get_new_feat(data[0][:,:,0:num_feat],flattened_list)
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
    results_32 = {}
    
    for data in dataloader:
        print('//////////////')
        other = data[3]
        print(other[0])

        num_feat = len(data[1].numpy()[0].tolist())
        print(num_feat)
        
        if num_feat<=32:
#        if num_feat == 32:
            flattened_list_32 = [x for x in range(0,num_feat)]
            updated_data = get_updated_data(data,flattened_list_32,num_feat)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_32[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
            results_16_16_64[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
            results_8_8_16_96[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
            results_4_4_8_16_128[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
            results_2_2_4_8_16_160[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
            results_1_1_2_4_8_16_192 = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)
        elif num_feat>32:
            flattened_list_32 = [x for x in range(0,32)]
            updated_data = get_updated_data(data,flattened_list_32,num_feat)
            outputs, loss, probs, _ = run_network(model, updated_data, 0, baseline)
            fps = outputs.size()[1]/other[1][0]
            gd_inf = updated_data[2]
            results_32[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], gd_inf.numpy()[0], fps)




    return results_32
#    return results_16_16_64 
#    return results_8_8_16_96
#    return results_4_4_8_16_128   
#    return results_2_2_4_8_16_160  
#    return results_1_1_2_4_8_16_192  
            

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
#        model = super_event.get_super_event_model(0, classes)
#        criterion = nn.NLLLoss(reduce=False)
#    
#        lr = 0.1*batch_size/len(datasets['train'])
#        print lr
#        optimizer = optim.Adam(model.parameters(), lr=lr)
#        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
#        
#        run([(model,0,dataloaders,optimizer, lr_sched, args.model_file)], criterion, num_epochs=40)

    else:
        print 'Evaluating...'
        rgb_model = torch.load(args.rgb_model_file)
        rgb_model.cuda()
        
        dataloaders, datasets = load_data('', test_split, rgb_root)
        
        flow_model = torch.load(args.flow_model_file)
        flow_model.cuda()
        
        dataloaders, datasets = load_data('', test_split, flow_root)
        
        start = time.clock()
        rgb_results = eval_model(rgb_model, dataloaders['val'], baseline=False)
        flow_results = eval_model(flow_model, dataloaders['val'], baseline=False)
        elapsed = (time.clock() - start)
        print("Time used:",elapsed)

        rapm = APMeter()
        fapm = APMeter()
        tapm = APMeter()


        for vid in rgb_results.keys():
            o,p,l,fps = rgb_results[vid]
            rapm.add(sigmoid(o), l)
            fapm.add(sigmoid(flow_results[vid][0]), l)
            if vid in flow_results:
                print('////////')
                print(vid)
                o2,p2,l2,fps = flow_results[vid]
                o = (o[:o2.shape[0]]*.5+o2*.5)
                p = (p[:p2.shape[0]]*.5+p2*.5)
            tapm.add(sigmoid(o), l)
            
        
        print 'rgb MAP:', rapm.value().mean()
        print 'flow MAP:', fapm.value().mean()
        print 'two-stream MAP:', tapm.value().mean()
