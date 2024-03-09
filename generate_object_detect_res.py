# Example
# python demo.py --net res101 --dataset vg --load_dir models --cuda
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from imageio import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
import pdb

import matplotlib.pyplot as plt
import json
from tqdm import tqdm  # 导入 tqdm 模块


xrange = range  # Python 3
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


conf_thresh = 0.4
MIN_BOXES = 10
MAX_BOXES = 36

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
      im_scale = float(target_size) / float(im_size_min)
      # Prevent the biggest axis from being more than MAX_SIZE
      if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
      im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
              interpolation=cv2.INTER_LINEAR)
      im_scale_factors.append(im_scale)
      processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

class Args():
    def __init__(self,dataset='vg',cfg='obj_detector/faster_r_cnn/cfgs/res101.yml',net='res101',load_dir='obj_detector/faster_r_cnn/data/faster_rcnn/',\
                 image_dir='obj_detector/faster_r_cnn/images',image_file='img0.jpg',classes_dir='obj_detector/faster_r_cnn/data/genome/1600-400-20',\
                    cuda=True,mGPUs=False,set=None,cag=False,parallel_type=0,vis=False):
        self.dataset=dataset
        self.cfg_file=cfg
        self.net=net
        self.load_dir=load_dir
        self.image_dir=image_dir
        self.image_file=image_file
        self.classes_dir=classes_dir
        self.cuda=cuda
        self.mGPUs=mGPUs
        self.set_cfgs=set
        self.class_agnostic=cag
        self.parallel_type=parallel_type
        self.vis=vis
args = Args(net='res101',dataset='vg')
# set cfg according to the dataset used to train the pre-trained model
if args.dataset == "pascal_voc":
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
elif args.dataset == "pascal_voc_0712":
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
elif args.dataset == "coco":
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
elif args.dataset == "imagenet":
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
elif args.dataset == "vg":
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

cfg.USE_GPU_NMS = args.cuda

# print('Using config:')
# pprint.pprint(cfg)
np.random.seed(cfg.RNG_SEED)
        

# Load faster rcnn model
if not os.path.exists(args.load_dir):
    raise Exception('There is no input directory for loading network from ' + args.load_dir)
load_name = os.path.join(args.load_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.dataset))

# Load classes
classes = ['__background__']
with open(os.path.join(args.classes_dir, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())
        
fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
fasterRCNN.create_architecture()

# 检查是否有 CUDA 支持
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")
fasterRCNN.to(device)

print("load checkpoint %s" % (load_name))
# 加载检查点，确保 map_location 与您的设备设置一致
checkpoint = torch.load(load_name, map_location=device)
fasterRCNN.load_state_dict(checkpoint['model'])
fasterRCNN.eval()

# 如果你想检查模型是否完全在GPU上
all_on_gpu = all(param.device.type == 'cuda' for param in fasterRCNN.parameters())
if all_on_gpu:
    print("Model is on GPU.")
else:
    print("Model is not fully on GPU.")


from src.datasets.data_utils.image_ops import img_from_base64
def get_image(bytestring): 
    # output numpy array (T, C, H, W), channel is RGB, T = 1
    cv2_im = img_from_base64(bytestring)
    cv2_im = cv2_im[:,:,::-1] # COLOR_BGR2RGB
    # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    output = np.transpose(cv2_im[np.newaxis, ...], (0, 3, 1, 2))
    return output

def detect(
    model,
    raw_img,
    classes,
    device=torch.device("cuda"),
    thresh=0.05
):
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)


    im_in = np.array(raw_img)
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :, ::-1]
    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
    )

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    # 确保所有输入数据都转移到了正确的设备
    im_data = im_data.to(device)
    im_info = im_info.to(device)
    gt_boxes = gt_boxes.to(device)
    num_boxes = num_boxes.to(device)
    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = (
                        box_deltas.view(-1, 4)
                        * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS
                    ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = (
                        box_deltas.view(-1, 4)
                        * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS
                    ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    max_conf = torch.zeros((pred_boxes.shape[0]))
    if args.cuda > 0:
        max_conf = max_conf.cuda()
    for j in xrange(1, len(classes)):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            index = inds[order[keep]]
            max_conf[index] = torch.where(
                scores[index, j] > max_conf[index], scores[index, j], max_conf[index]
            )
    keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0))
    keep_boxes = torch.squeeze(torch.nonzero(keep_boxes))

    # 使用 numel() 获取张量中的元素数量
    num_boxes = keep_boxes.numel()

    if num_boxes < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
    elif num_boxes > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]

    res = []
    boxes = pred_boxes[keep_boxes]
    objects = torch.argmax(scores[keep_boxes][:, 1:], dim=1)
    boxes_cpu = boxes.cpu()
    objects_cpu = objects.cpu()
    for i in range(len(keep_boxes)):
        kind = objects_cpu[i] + 1
        bbox = boxes_cpu[i, kind * 4 : (kind + 1) * 4].numpy()  # 现在将其转换为 NumPy 数组
        # 适当的修改，确保其他数据也在 CPU 上并转换为 NumPy 数组
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects_cpu[i] + 1]

        temp = dict()
        temp["cls"] = cls
        temp["bbox"] = bbox
        res.append(temp)
    return res

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def generate_obj_results(frames_dir='datasets/MSRVTT-v2/32frames/',
                          save_dir='datasets/MSRVTT-v2/objects/32frames/faster_rcnn/'):
    for i in tqdm(range(0, 10000), desc='Processing Videos'):  # 使用 tqdm 包装循环，显示进度条
        video_name = "video" + str(i) + "_frame"
        results = dict()
        for j in range(1, 33):
            if j <= 9:
                img_file = video_name + "000" + str(j) + ".jpg"
            elif j > 9 and j < 33:
                img_file = video_name + "00" + str(j) + ".jpg"
            if os.path.exists(frames_dir + img_file):
                raw_img = imread(frames_dir + img_file)
                res = detect(model=fasterRCNN, raw_img=raw_img, classes=classes)
                # 将NumPy数组转换为Python列表
                res = [{k: convert_to_json_serializable(v) for k, v in item.items()} for item in res]
                results[str(j)] = res
            else:
                results[str(j)] = None
        json_name = "video" + str(i) + ".json"
        with open(save_dir + json_name, 'w') as f:
            json.dump(results, f, default=convert_to_json_serializable)

generate_obj_results()

