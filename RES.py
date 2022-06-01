import argparse
import cv2
import os
import numpy as np
import json
from os import walk
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import random
import math
import shutil
import time

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif "imp" in name.lower():
                continue
            else:
                x = module(x)

        return target_activations, x


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_iou(x, y):
    intersection = np.bitwise_and(x, y)
    union = np.bitwise_or(x, y)

    iou = np.sum(intersection) / np.sum(union)

    return iou

def compute_exp_score(x, y):
    N =  np.sum(y!=0)
    epsilon = 1e-6
    tp = np.sum( x * (y>0))
    tn = np.sum((1-x) * (y<0))
    fp = np.sum( x * (y<0))
    fn = np.sum((1-x) * (y>0))

    exp_precision = tp / (tp + fp + epsilon)
    exp_recall = tp / (tp + fn + epsilon)
    exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall + epsilon)

    return exp_precision, exp_recall, exp_f1


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img,mask,path,file_name):
    save_path = os.path.join(path, file_name)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))


def resize_attention_label(path_to_attn, width=7, height=7):
    path_to_attn_resized = {}
    for img_path, img_att in path_to_attn.items():
        att_map = np.uint8(img_att) * 255
        img_att_resized = cv2.resize(att_map, (width, height), interpolation=cv2.INTER_NEAREST)

        path_to_attn_resized[img_path] = np.float32(img_att_resized / 255)
    return path_to_attn_resized


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sample_selection_with_explanations_gender(n_smaple_with_label, path_to_attn, label_ratio = 1):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_men = {}
    path_to_attn_women = {}
    source_dir_path = './gender_data/train'
    # before selection, let's create two pools for men and women separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/men/' + path):
            path_to_attn_men[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/women/' + path):
            path_to_attn_women[path] = path_to_attn[path]
        # else:
        #     print('Something wrong with this image:', path)

    print('Total number of explanation labels in train set - men:', len(path_to_attn_men))
    print('Total number of explanation labels in train set - women:', len(path_to_attn_women))
    random.seed(args.random_seed)
    sample_paths_men = random.sample(list(path_to_attn_men), n_smaple_with_label)
    sample_paths_women = random.sample(list(path_to_attn_women), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_men:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_women:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './gender_data/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/men')
        os.mkdir(fw_dir_path + '/women')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/men/' + path):
                src = source_dir_path + '/men/' + path
                dst = fw_dir_path + '/men/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/women/' + path):
                src = source_dir_path + '/women/' + path
                dst = fw_dir_path + '/women/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)


def sample_selection_with_explanations_places(n_smaple_with_label, path_to_attn, label_ratio = 1):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_nature = {}
    path_to_attn_urban = {}
    source_dir_path = './places/train'
    # before selection, let's create two pools for nature and urban separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/nature/' + path):
            path_to_attn_nature[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/urban/' + path):
            path_to_attn_urban[path] = path_to_attn[path]
        else:
            print('Something wrong with this image:', path)

    print('Total number of explanation labels in train set - nature:', len(path_to_attn_nature))
    print('Total number of explanation labels in train set - urban:', len(path_to_attn_urban))
    random.seed(args.random_seed)
    sample_paths_nature = random.sample(list(path_to_attn_nature), n_smaple_with_label)
    sample_paths_urban = random.sample(list(path_to_attn_urban), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_nature:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_urban:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './places/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/nature')
        os.mkdir(fw_dir_path + '/urban')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/nature/' + path):
                src = source_dir_path + '/nature/' + path
                dst = fw_dir_path + '/nature/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/urban/' + path):
                src = source_dir_path + '/urban/' + path
                dst = fw_dir_path + '/urban/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)


def load_path_to_attentions():
    # csv file format: img_idx,attention,img_check,matrix_resize
    path_to_attn = {}
    path_to_pos_attn = {}
    path_to_neg_attn = {}
    source_path = os.path.join(args.data_dir, 'attention_label')
    fns = next(walk(source_path), (None, None, []))[2]

    for fn in fns:
        df = pd.read_csv(os.path.join(source_path, fn))

        if 'counterfactual' in fn:
            # negative attention labels
            for index, row in df.iterrows():
                if row['img_check'] == 'good':
                    img_fn = row['img_idx'] + '.jpg'
                    path_to_neg_attn[img_fn] = np.array(json.loads(row['attention']))
        else:
            # positive attention labels
            for index, row in df.iterrows():
                if row['img_check'] == 'good':
                    img_fn = row['img_idx'] + '.jpg'
                    path_to_pos_attn[img_fn] = np.array(json.loads(row['attention']))
                    path_to_attn[img_fn] = ''

    for img_fn in path_to_attn.keys():
        if img_fn in path_to_pos_attn:
            if img_fn in path_to_neg_attn:
                path_to_attn[img_fn] = path_to_pos_attn[img_fn] - path_to_neg_attn[img_fn]
            else:
                path_to_attn[img_fn] = path_to_pos_attn[img_fn]

    # resized
    path_to_attn_resized = {}
    path_to_pos_attn = resize_attention_label(path_to_pos_attn)
    path_to_neg_attn = resize_attention_label(path_to_neg_attn)

    for img_fn in path_to_attn.keys():
        if img_fn in path_to_pos_attn:
            if img_fn in path_to_neg_attn:
                path_to_attn_resized[img_fn] = path_to_pos_attn[img_fn] - path_to_neg_attn[img_fn]
            else:
                path_to_attn_resized[img_fn] = path_to_pos_attn[img_fn]

    return path_to_attn, path_to_attn_resized


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ImageFolderWithMaps(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMaps, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # Add pred_weight & att_weight here
        if tail in path_to_attn_resized:
            true_attention_map = path_to_attn_resized[tail]
        else:
            true_attention_map = np.zeros((7, 7), dtype=np.float32)

        tuple_with_map = (original_tuple + (true_attention_map,))
        return tuple_with_map


class ImageFolderWithMapsAndWeights(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMapsAndWeights, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # default weights
        pred_weight = 1
        att_weight = 0
        # Add pred_weight & att_weight here
        if tail in path_to_attn_resized:
            true_attention_map_org = path_to_attn[tail]
            true_attention_map = path_to_attn_resized[tail]
            pred_weight = 1
            att_weight = 1

        else:
            true_attention_map_org = None
            true_attention_map = np.zeros((7, 7), dtype=np.float32)

        tuple_with_map_and_weights = (original_tuple + (true_attention_map, true_attention_map_org, pred_weight, att_weight))
        return tuple_with_map_and_weights


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='Number of epoch to run')
    parser.add_argument('--data_dir', default='gender_data', type=str)
    parser.add_argument('--model_dir', type=str, default='./model_save/',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--model_name', type=str, default='model_out',
                        help='The model filename that will be used for evaluation or phase 2 fine-tuning.')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('-ea', '--evaluate_all', dest='evaluate_all', action='store_true',
                        help='evaluate all models stored in model_dir')
    parser.add_argument('--trainWithMap', dest='trainWithMap', action='store_true',
                        help='train with edited attention map')
    parser.add_argument('--attention_weight', default=1.0, type=float,
                        help='Scale factor that balance between task loss and attention loss')
    parser.add_argument('--transforms', type=str, default=None,
                        help='The transform method to prcoess the human label, choices [None, gaussian, S1, S2, D1, D2]')
    parser.add_argument('--area', dest='area', action='store_true',
                        help='If only apply explanation loss to human labeled regions')
    parser.add_argument('--a', default=0.75, type=float,
                        help='Threshold  for function U')
    parser.add_argument('--eta', default=0.0, type=float,
                        help='Slack factor for robust attention loss')
    parser.add_argument('--fw_sample', default=0, type=int,
                        help='if non-zero, randomly sample instances to construct the dataset and perform learning')
    parser.add_argument('--random_seed', default=0, type=int, metavar='N',
                        help='random seed for sampling the dataset')
    parser.add_argument('--reg', default=0.0, type=float,
                        help='The scale factor for L2 regularization for deep imputation')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    # if args.use_cuda:
    #     print("Using GPU for acceleration")
    # else:
    #     print("Using CPU for computation")

    return args


args = get_args()
path_to_attn, path_to_attn_resized = load_path_to_attentions()
# resize attention label from 224x224 to 14x14
# path_to_attn_resized = resize_attention_label(path_to_attn)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        # self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def get_attention_map(self, input, index=None, norm = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]

        target = features[-1].squeeze()

        weights = torch.mean(grads_val, axis=(2, 3)).squeeze()

        if self.cuda:
            cam = torch.zeros(target.shape[1:]).cuda()
        else:
            cam = torch.zeros(target.shape[1:])

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        if norm == 'ReLU':
            cam = torch.relu(cam)
            cam = cam / (torch.max(cam) + 1e-6)
        elif norm == 'Sigmoid':
            cam = torch.sigmoid(cam)
        else:
            # linear (the original way used by GRADIA)
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-6)

        # need further investment about whether to normalize by max during training
        # cam = cam / (torch.max(cam) + 1e-6)

        # cam = torch.sigmoid(10*(cam-0.5))

        return cam, output

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            # print('model is looking at class', index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # use when visualizing explanation
        cam = cam - np.min(cam)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam / (np.max(cam) + 1e-6)

        # cam = 1 / (1 + np.exp(-10 * (cam - 0.5)))

        return cam


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def model_test(model, test_loader, output_attention=False, output_iou=False):
    # model.eval()
    iou = AverageMeter()
    exp_precision = AverageMeter()
    exp_recall = AverageMeter()
    exp_f1 = AverageMeter()
    ious = {}
    st = time.time()
    outputs_all = []
    targets_all = []
    img_fns = []

    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    y_label = np.array([])
    y_predict = np.array([])
    misclassified = np.array([])

    for batch_idx, (inputs, targets, paths) in enumerate(test_loader):
        y_label = np.append(y_label, targets)
        misclassified = np.append(misclassified, paths)

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            logits = model(inputs)
            outputs = torch.nn.functional.softmax(logits, dim=1)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            y_predict = np.append(y_predict, predicted)

        if output_attention:
            for img_path in paths:
                _, img_fn = os.path.split(img_path)

                img_fns.append(img_fn)

                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)
                show_cam_on_image(img, mask, 'attention', img_path)

                if output_iou and img_fn in path_to_attn:
                    item_att_binary = (mask > 0.5)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    iou.update(single_iou.item(), 1)

                    p, r, f1 = compute_exp_score(item_att_binary, target_att)
                    exp_precision.update(p.item(), 1)
                    exp_recall.update(r.item(), 1)
                    exp_f1.update(f1.item(), 1)

                    ious[img_fn] = single_iou.item()

        outputs_all += [outputs]
        targets_all += [targets]

    et = time.time()
    test_time = et - st

    test_acc = accuracy(torch.cat(outputs_all, dim=0), torch.cat(targets_all))[0].cpu().detach()

    return test_acc, iou.avg, exp_precision.avg, exp_recall.avg, exp_f1.avg


def BF_solver(X, Y):
    epsilon = 1e-4

    with torch.no_grad():
        x = torch.flatten(X)
        y = torch.flatten(Y)
        g_idx = (y<0).nonzero(as_tuple=True)[0]
        le_idx = (y>0).nonzero(as_tuple=True)[0]
        len_g = len(g_idx)
        len_le = len(le_idx)
        a = 0
        a_ct = 0.0
        for idx in g_idx:
            v = x[idx] + epsilon # to avoid miss the constraint itself
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

        for idx in le_idx:
            v = x[idx]
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

    # print('optimal solution for batch, a=', a)
    # print('final threshold a is assigned as:', am)

    return torch.tensor([a]).cuda()

def model_train_with_map(model, train_loader, val_loader, transforms = None, area = False, eta = 0.0):
    eta = torch.tensor([eta]).cuda()
    reg_criterion = nn.MSELoss()
    # reg_criterion = nn.L1Loss()
    BCE_criterion = nn.BCELoss()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    # load grad_cam module
    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight) in enumerate(train_loader):
            attention_loss = 0
            if args.use_cuda:
                inputs, targets, target_maps, target_maps_org, pred_weight, att_weight = inputs.cuda(), targets.cuda(
                    non_blocking=True), target_maps.cuda(), target_maps_org.cuda(), pred_weight.cuda(), att_weight.cuda()
            att_maps = []
            att_map_labels = []
            att_map_labels_trans = []
            att_weights = []
            outputs = model(inputs)

            for input, target, target_map, target_map_org, valid_weight in zip(inputs, targets, target_maps, target_maps_org, att_weight):
                # only train on img with attention labels
                if valid_weight > 0.0:
                    # get attention maps from grad-CAM
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target, norm = None)
                    att_maps.append(att_map)

                    if transforms == 'Gaussian':
                        # here we only work on positive labels for D loss
                        target_map_pos = np.maximum(target_map.cpu().numpy(), 0)
                        target_map_trans = cv2.GaussianBlur(target_map_pos, (3, 3), 0)
                        target_map_trans = target_map_trans / (np.max(target_map_trans)+1e-6)
                        att_map_labels_trans.append(torch.from_numpy(target_map_trans).cuda())
                    elif transforms == 'S1':
                        target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                        input_imp = target_map_pos_org

                        target_map_trans = model.imp(torch.unsqueeze(input_imp, 0))
                        temp = torch.squeeze(target_map_trans)
                        temp = temp - torch.min(temp)
                        temp = temp / (torch.max(temp) + 1e-6)
                        att_map_labels_trans.append(temp)
                    elif transforms == 'S2':
                        target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                        # with both input X and the human mask F (input 3x224x224, we need the raw target map in 1x224x224)
                        input_imp = torch.cat((target_map_pos_org, input), 0)

                        target_map_trans = model.imp(torch.unsqueeze(input_imp, 0))
                        temp = torch.squeeze(target_map_trans)
                        temp = temp - torch.min(temp)
                        temp = temp / (torch.max(temp) + 1e-6)
                        att_map_labels_trans.append(temp)
                    elif transforms == 'D1':
                        target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                        input_imp = target_map_pos_org

                        H1 = torch.relu(model.imp_conv1(torch.unsqueeze(input_imp, 0)))
                        H2 = torch.relu(model.imp_conv2(H1))
                        H3 = torch.relu(model.imp_conv3(H2))
                        H4 = torch.relu(model.imp_conv4(H3))
                        H5 = model.imp_conv5(H4)

                        temp = torch.squeeze(H5)
                        temp = temp - torch.min(temp)
                        temp = temp / (torch.max(temp) + 1e-6)
                        att_map_labels_trans.append(temp)
                    elif transforms == 'D2':
                        target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                        # 4x224x224
                        input_imp = torch.cat((target_map_pos_org, input), 0)

                        H1 = torch.relu(model.imp_conv1(torch.unsqueeze(input_imp, 0)))
                        H2 = torch.relu(model.imp_conv2(H1))
                        H3 = torch.relu(model.imp_conv3(H2))
                        H4 = torch.relu(model.imp_conv4(H3))
                        H5 = model.imp_conv5(H4)

                        temp = torch.squeeze(H5)
                        temp = temp - torch.min(temp)
                        temp = temp / (torch.max(temp) + 1e-6)
                        att_map_labels_trans.append(temp)

                    att_map_labels.append(target_map)
                    att_weights.append(valid_weight)

            # compute task loss
            task_loss = task_criterion(outputs, targets)
            task_loss = torch.mean(pred_weight * task_loss)

            # compute exp loss
            if att_maps:
                att_maps = torch.stack(att_maps)
                att_map_labels = torch.stack(att_map_labels)

                if transforms == 'S1' or transforms == 'S2' or transforms == 'D1' or transforms == 'D2' or transforms == 'Gaussian':
                    # hard threshold solver for a
                    a = BF_solver(att_maps, att_map_labels)
                    # alternatively, we can use tanh as surrogate loss to make att_maps trainable
                    temp1 = torch.tanh(5*(att_maps - a))
                    temp_loss = attention_criterion(temp1, att_map_labels)

                    # normalize by effective areas
                    temp_size = (att_map_labels != 0).float()
                    eff_loss = torch.sum(temp_loss * temp_size) / torch.sum(temp_size)
                    attention_loss += torch.relu(torch.mean(eff_loss) - eta)
                else:
                    a = 0

                if transforms == 'S1' or transforms == 'S2' or transforms == 'D1' or transforms == 'D2':
                    att_map_labels_trans = torch.stack(att_map_labels_trans)
                    tempD = attention_criterion(att_maps, att_map_labels_trans)
                    # regularization (currently prefer not use it)
                    reg_loss = reg_criterion(att_map_labels_trans, att_map_labels * (att_map_labels > 0).float())
                    attention_loss += args.reg * reg_loss
                elif transforms == 'Gaussian':
                    att_map_labels_trans = torch.stack(att_map_labels_trans)
                    tempD = attention_criterion(att_maps, att_map_labels_trans)
                elif transforms == 'HAICS':
                    tempD = BCE_criterion(att_maps, att_map_labels * (att_map_labels > 0).float()) * (att_map_labels != 0).float()
                else: # GRADIA
                    tempD = attention_criterion(att_maps, att_map_labels * (att_map_labels > 0).float())

                attention_loss += torch.mean(tempD)
                loss = task_loss + attention_loss
            else:
                loss = task_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all += [outputs]
            targets_all += [targets]

            print('Batch_idx :', batch_idx, ', task_loss', task_loss, ', attention_loss', attention_loss, ', a:', a)
            # print('Batch_idx :', batch_idx, ', task_loss:', task_loss, ', attention_loss', 0.3*attention_loss, ', pos_loss:', torch.mean(pos_loss), ', neg_loss:', torch.mean(neg_loss), ', a:', a)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []

        iou = AverageMeter()
        for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = model(inputs)

            for img_path in paths:
                _, img_fn = os.path.split(img_path)
                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)

                if img_fn in path_to_attn:
                    item_att_binary = (mask > 0.5)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    iou.update(single_iou.item(), 1)

            outputs_all += [outputs]
            targets_all += [targets]

        et = time.time()
        test_time = et - st

        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()
        val_iou = iou.avg
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.model_dir, args.model_name))
            print('UPDATE!!!')

        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc, 'Val IOU:', val_iou)

    return best_val_acc


def model_train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_acc = 0

    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all += [outputs]
            targets_all += [targets]

            print('Batch_idx :', batch_idx, ', loss', loss)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)

                outputs_all += [outputs]
                targets_all += [targets]

        et = time.time()
        val_time = et - st
        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.model_dir, args.model_name))
            print('UPDATE!!!')

        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc)

    return best_val_acc

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Data loading code
    if args.fw_sample > 0:
        print('Performing few sample learning with # of samples = ', args.fw_sample)

        if args.data_dir == 'gender_data':
            sample_selection_with_explanations_gender(args.fw_sample, path_to_attn)
        elif args.data_dir == 'places':
            sample_selection_with_explanations_places(args.fw_sample, path_to_attn)
        else:
            print('Error: Unrecognized dataset:', args.data)

        traindir = os.path.join(args.data_dir, 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed))
    else:
        traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    testdir = os.path.join(args.data_dir, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.trainWithMap:
        train_with_map_loader = torch.utils.data.DataLoader(
            ImageFolderWithMapsAndWeights(traindir, transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.train_batch, shuffle=True,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.train_batch, shuffle=True,
            pin_memory=True)

    train_for_eval_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        pin_memory=True)

    # check target_layer_names when change resnet18 to resnet50 (18 should be [1] 50 should be [2])
    model = models.resnet50(pretrained=True)
    # replace the original output layer from 1000 classes to 2 class for man and woman task
    model.fc = nn.Linear(2048, 2)

    if args.transforms == 'S1':
        # shallow imputation without X
        model.imp = nn.Conv2d(1, 1, 64, stride=32, padding=16)
    elif args.transforms == 'S2':
        # shallow imputation with X as additional input
        model.imp = nn.Conv2d(4, 1, 64, stride=32, padding=16)
    elif args.transforms == 'D1':
        # deep imputation without X
        model.imp_conv1 = nn.Conv2d(1, 1, 7, stride=2, padding=3)
        model.imp_conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv3 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv4 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv5 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
    elif args.transforms == 'D2':
        # deep imputation with X as residual input
        model.imp_conv1 = nn.Conv2d(4, 4, 7, stride=2, padding=3)
        model.imp_conv2 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv3 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv4 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv5 = nn.Conv2d(4, 1, 3, stride=2, padding=1)
    model.a = nn.Parameter(torch.tensor([args.a]))

    if args.use_cuda:
        model.cuda()

    if args.evaluate_all:
        for (dirpath, dirnames, model_fns) in walk(args.model_dir):
            for model_fn in model_fns:
                model = torch.load(os.path.join(dirpath, model_fn))
                test_acc, test_iou, test_precision, test_recall, test_f1 = model_test(model, test_loader,
                                                                                                output_attention=True,
                                                                                                output_iou=True)
                print('Model:', model_fn, ', Acc:', test_acc, ', IOU:', test_iou,
                      ', P:', test_precision, ', R:', test_recall, ', F1:', test_f1)

    elif args.evaluate:
        # attention_shift_study(val_loader)
        model = torch.load(os.path.join(args.model_dir, args.model_name))

        # evaluate model on test set (set output_attention=true if you want to save the model generated attention)
        test_acc, test_iou, test_precision, test_recall, test_f1 = model_test(model, test_loader, output_attention=True, output_iou=True)
        print('Finish Testing. Test Acc:', test_acc, ', Test IOU:', test_iou, ', Test Precision:', test_precision, ', Test Recall:', test_recall, ', Test F1:', test_f1)
    else:
        if args.trainWithMap:
            # for ours
            print('Init training with explanation supervision..')
            best_val_acc = model_train_with_map(model, train_with_map_loader, val_loader, transforms=args.transforms, area=args.area, eta=args.eta)
            print('Finish Training. Best Validation acc:', best_val_acc)
        else:
            # for baseline without explanation supervision
            print('Init training for baseline..')
            best_val_acc = model_train(model, train_loader, val_loader)
            print('Finish Training. Best Validation acc:', best_val_acc)
