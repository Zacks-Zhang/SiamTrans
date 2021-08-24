# -*- coding: utf-8 -*

from loguru import logger

import torch
import torch.nn.functional as F
from torch import nn

import torchvision.utils as vutils

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
from videoanalyst.model.task_model.taskmodel_impl.transformer.featurefusion_network import FeatureFusionNetwork
from videoanalyst.model.task_model.taskmodel_impl.transformer.utils import build_position_encoding, \
    nested_tensor_from_tensor, nested_tensor_from_tensor_2, NestedTensor

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                neck_conv_bias=[True, True, True, True],
                                corr_fea_output=False,
                                trt_mode=False,
                                trt_fea_model_path="",
                                trt_track_model_path="",
                                amp=False,
                                use_transformer=False,
                                show_featuremap=False
                                )

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, head, loss=None):
        super(SiamTrack, self).__init__()
        self.basemodel = backbone
        self.head = head
        self.loss = loss
        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"


    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def train_forward(self, training_data):
        target_img = training_data["im_z"]
        search_img = training_data["im_x"]

        if self._hyper_params['use_transformer']:
            target_img_nested = nested_tensor_from_tensor(target_img)
            target_img = target_img_nested.tensors
            search_img_nested = nested_tensor_from_tensor(search_img)
            search_img = search_img_nested.tensors


        # backbone feature
        f_z = self.basemodel(target_img)
        f_x = self.basemodel(search_img)

        if self._hyper_params['use_transformer']:
            # transformer
            # mask
            mask_z = F.interpolate(target_img_nested.mask[None].float(), size=f_z.shape[-2:]).to(torch.bool)[0]
            mask_x = F.interpolate(search_img_nested.mask[None].float(), size=f_x.shape[-2:]).to(torch.bool)[0]
            # position encoding
            pos_z = []
            pos_z.append(self.pos_encoding(NestedTensor(f_z, mask_z)).to(f_z.dtype))
            pos_x = []
            pos_x.append(self.pos_encoding(NestedTensor(f_x, mask_x)).to(f_x.dtype))

            assert mask_z is not None
            assert mask_x is not None

            # 使用transformer进行特征融合的话
            # f_z和f_x会提前融合成一个特征图
            # 然后微调成cls和reg两个分支
            # 比原来少了两个分支

            # feature enhance and fuse
            f_fused = self.feature_fusion(self.input_proj(f_z), mask_z,
                                     self.input_proj(f_x), mask_x,
                                     pos_z[-1], pos_x[-1])  # [1, 2, 256, 625]

            f_fused = f_fused.permute(1, 2, 0, 3).reshape(f_x.shape)

            # 生成回归和分类分支特征
            # feature adjust
            c_out = self.conv_to_cls(f_fused)
            r_out = self.conv_to_reg(f_fused)

        else:
            # feature adjustment
            c_z_k = self.c_z_k(f_z)
            r_z_k = self.r_z_k(f_z)
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)

        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
            c_out, r_out)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea

        if self._hyper_params["show_featuremap"]:
            visualized_data = {}
            img_grids = vutils.make_grid(f_fused[0, :, :, :].unsqueeze(0).permute(1, 0, 2, 3), normalize=True, scale_each=True, nrow=4, padding=1)
            visualized_data["fused"] = img_grids
            visualized_data["origin_z"] = target_img
            visualized_data["origin_x"] = search_img
            return predict_data, visualized_data
        return predict_data

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            target_img, = args
            if self._hyper_params["trt_mode"]:
                # extract feature with trt model
                out_list = self.trt_fea_model(target_img)
            else:
                # backbone feature
                if self._hyper_params['use_transformer']:
                    target_img_nested = nested_tensor_from_tensor_2(target_img)
                    target_img = target_img_nested.tensors
                f_z = self.basemodel(target_img)

                if self._hyper_params['use_transformer']:
                    # transformer
                    # mask
                    mask_z = F.interpolate(target_img_nested.mask[None].float(), size=f_z.shape[-2:]).to(torch.bool)[0]
                    # position encoding
                    pos_z = []
                    pos_z.append(self.pos_encoding(NestedTensor(f_z, mask_z)).to(f_z.dtype))
                    out_list = [NestedTensor(f_z, mask_z), pos_z]
                else:
                    # template as kernel
                    c_z_k = self.c_z_k(f_z)
                    r_z_k = self.r_z_k(f_z)
                    # output
                    out_list = [c_z_k, r_z_k]
        # used for template feature extraction (trt mode)
        elif phase == "freeze_track_fea":
            search_img, = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # head
            return [c_x, r_x]
        # [Broken] used for template feature extraction (trt mode)
        #   currently broken due to following issue of "torch2trt" package
        #   c.f. https://github.com/NVIDIA-AI-IOT/torch2trt/issues/251
        elif phase == "freeze_track_head":
            c_out, r_out = args
            # head
            outputs = self.head(c_out, r_out, 0, True)
            return outputs
        # used for tracking one frame during test
        elif phase == 'track':
            if len(args) == 3:
                if self._hyper_params['use_transformer']:
                    search_img, f_z_nested, pos_z = args
                else:
                    search_img, c_z_k, r_z_k = args
                if self._hyper_params["trt_mode"]:
                    c_x, r_x = self.trt_track_model(search_img)
                else:
                    if self._hyper_params['use_transformer']:
                        search_img_nested = nested_tensor_from_tensor_2(search_img)
                        search_img = search_img_nested.tensors
                        f_z = f_z_nested.tensors

                    # backbone feature
                    f_x = self.basemodel(search_img)

                    if self._hyper_params['use_transformer']:
                        mask_z = f_z_nested.mask
                        mask_x = F.interpolate(search_img_nested.mask[None].float(), size=f_x.shape[-2:]).to(torch.bool)[0]
                        # position encoding
                        pos_x = []
                        pos_x.append(self.pos_encoding(NestedTensor(f_x, mask_x)).to(f_x.dtype))

                        assert mask_z is not None
                        assert mask_x is not None
                    else:
                        # feature adjustment
                        c_x = self.c_x(f_x)
                        r_x = self.r_x(f_x)
            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            if self._hyper_params['use_transformer']:
                # feature enhance and fuse
                f_fused = self.feature_fusion(self.input_proj(f_z), mask_z,
                                              self.input_proj(f_x), mask_x,
                                              pos_z[-1], pos_x[-1])  # [1, 2, 256, 625]

                f_fused = f_fused.permute(1, 2, 0, 3).reshape(f_x.shape)

                # 生成回归和分类分支特征
                # feature adjust
                c_out = self.conv_to_cls(f_fused)
                r_out = self.conv_to_reg(f_fused)
            else:
                # feature matching
                r_out = xcorr_depthwise(r_x, r_z_k)
                c_out = xcorr_depthwise(c_x, c_z_k)

            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            # register extra output
            if self._hyper_params['use_transformer']:
                extra = dict(corr_fea=f_fused)
            else:
                extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
                self.cf = c_x
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        self._initialize_conv()
        super().update_params()
        if self._hyper_params["trt_mode"]:
            logger.info("trt mode enable")
            from torch2trt import TRTModule
            self.trt_fea_model = TRTModule()
            self.trt_fea_model.load_state_dict(
                torch.load(self._hyper_params["trt_fea_model_path"]))
            self.trt_track_model = TRTModule()
            self.trt_track_model.load_state_dict(
                torch.load(self._hyper_params["trt_track_model_path"]))
            logger.info("loading trt model succefully")


        if self._hyper_params['use_transformer']:
            self.pos_encoding = build_position_encoding(d_model=256, position_embedding='sine')
            self.input_proj = nn.Conv2d(256, 256, kernel_size=1)  # 其实可以没有这个
            self.feature_fusion = FeatureFusionNetwork(d_model=256, nhead=8, num_featurefusion_layers=4,
                                                       dim_feedforward=2048, dropout=0.1)
            channels = self._hyper_params['head_width']
            self.conv_to_reg = conv_bn_relu(channels, channels, 1, 5, 0, has_relu=False)
            self.conv_to_cls = conv_bn_relu(channels, channels, 1, 5, 0, has_relu=False)


    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
