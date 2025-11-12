# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import time
import cv2
import torch
from hoi_lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from hoi_lib.model.rpn.bbox_transform import clip_boxes
from torchvision.ops import nms
from hoi_lib.model.rpn.bbox_transform import bbox_transform_inv
from hoi_lib.model.utils.net_utils import vis_detections_filtered_objects_PIL
from hoi_lib.model.utils.blob import im_list_to_blob
from hoi_lib.model.faster_rcnn.vgg16 import vgg16
from hoi_lib.model.faster_rcnn.resnet import resnet
import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--base_dir', dest='base_dir',
                        help='optional config file',
                        default='/home/dholzmann/Workspace/hand_object_detector', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results',
                        default="images_det")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=89999, type=int, required=True)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
xrange = range  # Python 3
pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
np.random.seed(cfg.RNG_SEED)


def _get_image_blob(im: np.ndarray):
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


class HOIDetector(torch.nn.Module):

    def __init__(self, base_dir, load_dir="models", net="res101", dataset="pascal_voc", class_agnostic=False, checksession=1, checkepoch=8, checkpoint=132028, device='cpu', ths_hand=0.5, ths_obj=0.5):
        super().__init__()
        self.model = load_model(base_dir, load_dir, net, dataset, class_agnostic, checksession, checkepoch, checkpoint, device)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.ths_hand = ths_hand
        self.ths_obj = ths_obj
        self.class_agnostic = class_agnostic

    def _fwd(self, im: np.ndarray):
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1).to(self.device)
        im_info = torch.FloatTensor(1).to(self.device)
        num_boxes = torch.LongTensor(1).to(self.device)
        gt_boxes = torch.FloatTensor(1).to(self.device)
        box_info = torch.FloatTensor(1)

        with torch.no_grad():
            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_()

            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, loss_list = self.model(im_data, im_info, gt_boxes, num_boxes, box_info)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0]  # hand contact state info
            offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

            # get hand contact
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS).to(self.device) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(self.device)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS).to(self.device) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(self.device)
                        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            obj_dets, hand_dets = np.zeros((1, 10)), np.zeros((1, 10))
            for j in xrange(1, len(pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:, j] > self.ths_hand).view(-1)
                elif pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:, j] > self.ths_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat(
                        (cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds],
                         lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()
        return obj_dets, hand_dets

    def forward(self, images: np.ndarray):
        obj_dets, hand_dets = [], []
        for image in images:
            o, h = self._fwd(image)
            o_bbox = np.round(o[:4]).astype(np.int32)
            h_bbox = np.round(h[:4]).astype(np.int32)
            obj_dets.append(o)
            hand_dets.append(h)
        return np.concatenate(obj_dets), np.concatenate(hand_dets)


def load_model(base_dir, load_dir, net, dataset, class_agnostic, checksession, checkepoch, checkpoint, device):
    model_dir = base_dir + "/" + load_dir + "/" + net + "_handobj_100K" + "/" + dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])

    # initilize the network here.
    if net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=class_agnostic)
    elif net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    return fasterRCNN



if __name__ == '__main__':

    args = parse_args()

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)
    model = HOIDetector(args.base_dir, args.load_dir, args.net, args.dataset, args.class_agnostic, args.checksession, args.checkepoch, args.checkpoint, "cuda", args.thresh_hand, args.thresh_obj)

    vis = args.vis
    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        print(f'image dir = {args.base_dir + "/" + args.image_dir}')
        print(f'save dir = {args.base_dir + "/" + args.save_dir}')
        imglist = os.listdir(args.base_dir + "/" + args.image_dir)
        num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))
    imgs = []
    while (num_images >= 0):
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            im_in = cv2.imread(im_file)
        # bgr
        im = im_in
        imgs.append(im)

    obj_dets_list, hand_dets_list = model(imgs)
    im2show = imgs[0]
    obj_dets, hand_dets = obj_dets_list[0], hand_dets_list[0]

    if vis:
        # visualization
        im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, args.thresh_hand, args.thresh_obj)

    if vis and webcam_num == -1:

        folder_name = args.save_dir
        os.makedirs(folder_name, exist_ok=True)
        result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
        im2show.save(result_path)
    else:
        im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", im2showRGB)
        cv2.waitKey(1)

    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()