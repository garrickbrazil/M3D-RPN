import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch


def dilate_layer(layer, val):

    layer.dilation = val
    layer.padding = val


class RPN(nn.Module):


    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.base = base

        del self.base.transition3.pool

        # dilate
        dilate_layer(self.base.denseblock4.denselayer1.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer2.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer3.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer4.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer5.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer6.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer7.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer8.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer9.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer10.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer11.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer12.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer13.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer14.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer15.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer16.conv2, 2)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors


    def forward(self, x):

        batch_size = x.size(0)

        # resnet
        x = self.base(x)

        prop_feats = self.prop_feats(x)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'

    densenet121 = models.densenet121(pretrained=train)

    rpn_net = RPN(phase, densenet121.features, conf)

    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net
