import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch.nn.functional as F
import torch


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.group_conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows)

    def forward(self, x):

        b, c, h, w = x.size()

        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        t = int(h / self.num_rows)

        # unfold by rows
        x = x.unfold(2, t + self.pad*2, t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad*2, (w+self.pad*2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)

        return y


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

        self.num_rows = int(min(conf.bins, calc_output_size(conf.test_scale, conf.feat_stride)))

        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1, )

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

        self.prop_feats_loc = nn.Sequential(
            LocalConv2d(self.num_rows, self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # outputs
        self.cls_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1, )

        # bbox 2d
        self.bbox_x_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.cls_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))

        self.bbox_x_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_y_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_w_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_h_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))

        self.bbox_x3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_y3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_z3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_w3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_h3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_l3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.bbox_rY3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds

    def forward(self, x):

        batch_size = x.size(0)

        # resnet
        x = self.base(x)

        prop_feats = self.prop_feats(x)
        prop_feats_loc = self.prop_feats_loc(x)

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

        cls_loc = self.cls_loc(prop_feats_loc)

        # bbox 2d
        bbox_x_loc = self.bbox_x_loc(prop_feats_loc)
        bbox_y_loc = self.bbox_y_loc(prop_feats_loc)
        bbox_w_loc = self.bbox_w_loc(prop_feats_loc)
        bbox_h_loc = self.bbox_h_loc(prop_feats_loc)

        # bbox 3d
        bbox_x3d_loc = self.bbox_x3d_loc(prop_feats_loc)
        bbox_y3d_loc = self.bbox_y3d_loc(prop_feats_loc)
        bbox_z3d_loc = self.bbox_z3d_loc(prop_feats_loc)
        bbox_w3d_loc = self.bbox_w3d_loc(prop_feats_loc)
        bbox_h3d_loc = self.bbox_h3d_loc(prop_feats_loc)
        bbox_l3d_loc = self.bbox_l3d_loc(prop_feats_loc)
        bbox_rY3d_loc = self.bbox_rY3d_loc(prop_feats_loc)

        cls_ble = self.sigmoid(self.cls_ble)

        # bbox 2d
        bbox_x_ble = self.sigmoid(self.bbox_x_ble)
        bbox_y_ble = self.sigmoid(self.bbox_y_ble)
        bbox_w_ble = self.sigmoid(self.bbox_w_ble)
        bbox_h_ble = self.sigmoid(self.bbox_h_ble)

        # bbox 3d
        bbox_x3d_ble = self.sigmoid(self.bbox_x3d_ble)
        bbox_y3d_ble = self.sigmoid(self.bbox_y3d_ble)
        bbox_z3d_ble = self.sigmoid(self.bbox_z3d_ble)
        bbox_w3d_ble = self.sigmoid(self.bbox_w3d_ble)
        bbox_h3d_ble = self.sigmoid(self.bbox_h3d_ble)
        bbox_l3d_ble = self.sigmoid(self.bbox_l3d_ble)
        bbox_rY3d_ble = self.sigmoid(self.bbox_rY3d_ble)

        # blend
        cls = (cls * cls_ble) + (cls_loc * (1 - cls_ble))

        bbox_x = (bbox_x * bbox_x_ble) + (bbox_x_loc * (1 - bbox_x_ble))
        bbox_y = (bbox_y * bbox_y_ble) + (bbox_y_loc * (1 - bbox_y_ble))
        bbox_w = (bbox_w * bbox_w_ble) + (bbox_w_loc * (1 - bbox_w_ble))
        bbox_h = (bbox_h * bbox_h_ble) + (bbox_h_loc * (1 - bbox_h_ble))

        bbox_x3d = (bbox_x3d * bbox_x3d_ble) + (bbox_x3d_loc * (1 - bbox_x3d_ble))
        bbox_y3d = (bbox_y3d * bbox_y3d_ble) + (bbox_y3d_loc * (1 - bbox_y3d_ble))
        bbox_z3d = (bbox_z3d * bbox_z3d_ble) + (bbox_z3d_loc * (1 - bbox_z3d_ble))
        bbox_h3d = (bbox_h3d * bbox_h3d_ble) + (bbox_h3d_loc * (1 - bbox_h3d_ble))
        bbox_w3d = (bbox_w3d * bbox_w3d_ble) + (bbox_w3d_loc * (1 - bbox_w3d_ble))
        bbox_l3d = (bbox_l3d * bbox_l3d_ble) + (bbox_l3d_loc * (1 - bbox_l3d_ble))
        bbox_rY3d = (bbox_rY3d * bbox_rY3d_ble) + (bbox_rY3d_loc * (1 - bbox_rY3d_ble))

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

        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
            self.rois = self.rois.type(torch.cuda.FloatTensor)

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):
    train = phase.lower() == 'train'

    densenet121 = models.densenet121(pretrained=train)

    num_cls = len(conf['lbls']) + 1
    num_anchors = conf['anchors'].shape[0]

    # make network
    rpn_net = RPN(phase, densenet121.features, conf)

    dst_weights = rpn_net.state_dict()
    dst_keylist = list(dst_weights.keys())

    if 'pretrained' in conf and conf.pretrained is not None:

        src_weights = torch.load(conf.pretrained)
        src_keylist = list(src_weights.keys())

        conv_layers = ['cls', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'bbox_x3d', 'bbox_y3d', 'bbox_w3d', 'bbox_h3d', 'bbox_l3d', 'bbox_z3d', 'bbox_rY3d']

        # prop_feats
        src_weight_key = 'module.{}.0.weight'.format('prop_feats')
        src_bias_key = 'module.{}.0.bias'.format('prop_feats')

        dst_weight_key = 'module.{}.0.group_conv.weight'.format('prop_feats_loc')
        dst_bias_key = 'module.{}.0.group_conv.bias'.format('prop_feats_loc')

        src_weights[dst_weight_key] = src_weights[src_weight_key].repeat(conf.bins, 1, 1, 1)
        src_weights[dst_bias_key] = src_weights[src_bias_key].repeat(conf.bins, )

        #del src_weights[src_weight_key]
        #del src_weights[src_bias_key]

        for layer in conv_layers:
            src_weight_key = 'module.{}.weight'.format(layer)
            src_bias_key = 'module.{}.bias'.format(layer)

            dst_weight_key = 'module.{}.group_conv.weight'.format(layer+'_loc')
            dst_bias_key = 'module.{}.group_conv.bias'.format(layer+'_loc')

            src_weights[dst_weight_key] = src_weights[src_weight_key].repeat(conf.bins, 1, 1, 1)
            src_weights[dst_bias_key] = src_weights[src_bias_key].repeat(conf.bins, )

            #del src_weights[src_weight_key]
            #del src_weights[src_bias_key]

        src_keylist = list(src_weights.keys())

        # rename layers without module..
        for key in src_keylist:

            key_new = key.replace('module.', '')
            if key_new in dst_keylist:
                src_weights[key_new] = src_weights[key]
                del src_weights[key]

        src_keylist = list(src_weights.keys())

        # new layers?
        for key in dst_keylist:
            if key not in src_keylist:
                src_weights[key] = dst_weights[key]

        # remove keys not in dst
        for key in src_keylist:
            if key not in dst_keylist:
                del src_weights[key]

        rpn_net.load_state_dict(src_weights)

    if train:
        rpn_net.train()
    else:
        rpn_net.eval()

    return rpn_net
