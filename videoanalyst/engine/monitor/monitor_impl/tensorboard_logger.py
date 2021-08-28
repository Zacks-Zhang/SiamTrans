# -*- coding: utf-8 -*
import os.path as osp
from collections import Mapping
from typing import Dict

import cv2
from loguru import logger

from torch.utils.tensorboard.writer import SummaryWriter

from videoanalyst.utils import ensure_dir

from ..monitor_base import TRACK_MONITORS, VOS_MONITORS, MonitorBase

import numpy as np
from skimage import io

@TRACK_MONITORS.register
@VOS_MONITORS.register
class TensorboardLogger(MonitorBase):
    r"""Log training info to tensorboard for better visualization

    Hyper-parameters
    ----------------
    exp_name : str
        experiment name
    exp_save : str
        directory to save snapshots
    log_dir : str
        places to save tensorboard file
        will be updated in update_params
        EXP_SAVE/EXP_NAME/logs/tensorboard 
    """

    default_hyper_params = dict(
        exp_name="",
        exp_save="",
        log_dir="",
    )

    def __init__(self, ):
        r"""
        Arguments
        ---------
        """
        super(TensorboardLogger, self).__init__()
        self._state["writer"] = None

    def update_params(self):
        self._hyper_params["log_dir"] = osp.join(
            self._hyper_params["exp_save"],
            self._hyper_params["exp_name"],
            "logs/tensorboard",
        )

    def init(self, engine_state: Dict):
        super(TensorboardLogger, self).init(engine_state)

    def update(self, engine_data: Dict):
        # from engine state calculate global step
        engine_state = self._state["engine_state"]
        epoch = engine_state["epoch"]
        max_epoch = engine_state["max_epoch"]
        iteration = engine_state["iteration"]
        max_iteration = engine_state["max_iteration"]
        global_step = iteration + epoch * max_iteration

        # build at first update
        if self._state["writer"] is None:
            self._build_writer(global_step=global_step)
            logger.info(
                "Tensorboard writer built, starts recording from global_step=%d"
                % global_step, )
            logger.info(
                "epoch=%d, max_epoch=%d, iteration=%d, max_iteration=%d" %
                (epoch, max_epoch, iteration, max_iteration))
        writer = self._state["writer"]

        # traverse engine_data and put to scalar
        self._add_scalar_recursively(writer, engine_data, "", global_step)

    def update_pic(self, pic_data):
        # from engine state calculate global step
        engine_state = self._state["engine_state"]
        epoch = engine_state["epoch"]
        max_epoch = engine_state["max_epoch"]
        iteration = engine_state["iteration"]
        max_iteration = engine_state["max_iteration"]
        global_step = iteration + epoch * max_iteration

        # build at first update
        if self._state["writer"] is None:
            self._build_writer(global_step=global_step)
            logger.info(
                "Tensorboard writer built, starts recording from global_step=%d"
                % global_step, )
            logger.info(
                "epoch=%d, max_epoch=%d, iteration=%d, max_iteration=%d" %
                (epoch, max_epoch, iteration, max_iteration))
        writer = self._state["writer"]

        heatmap = cv2.applyColorMap(np.uint8(255 * pic_data[0]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]  # gbr to rgb

        # 合并heatmap到原始图像
        # cam = heatmap + np.float32(pic_data[1].permute(1,2,0).cpu())
        heatmap = (heatmap * 255).astype(np.uint8)
        # cam -= np.max(np.min(cam), 0)
        # cam /= np.max(cam)
        # cam = (cam * 255.).astype(np.uint8)

        # io.imsave("test.jpg", cam)
        writer.add_image("origin", pic_data[1], global_step=global_step, dataformats="CHW")
        writer.add_image("featmap", heatmap, global_step=global_step, dataformats="HWC")
        # traverse engine_data and put to scalar
        # writer.add_image("fused feature", pic_data["fused"], global_step=global_step, dataformats="CHW")
        # writer.add_image("origin template pic", pic_data["origin_z"], global_step=global_step, dataformats="NCHW")
        # writer.add_image("origin search pic", pic_data["origin_x"], global_step=global_step, dataformats="NCHW")


    def _build_writer(self, global_step=0):
        log_dir = self._hyper_params["log_dir"]
        ensure_dir(log_dir)
        self._state["writer"] = SummaryWriter(
            log_dir=log_dir,
            purge_step=global_step,
            filename_suffix="",
        )

    def _add_scalar_recursively(self, writer: SummaryWriter, o, prefix: str,
                                global_step: int):
        """Recursively add scalar from mapping-like o: tag1/tag2/tag3/...
        
        Parameters
        ----------
        writer : SummaryWriter
            writer
        o : mapping-like or scalar
            [description]
        prefix : str
            tag prefix, tag is the name to be passed into writer
        global_step : int
            global step counter
        """
        if isinstance(o, Mapping):
            for k in o:
                if len(prefix) > 0:
                    tag = "%s/%s" % (prefix, k)
                else:
                    tag = k
                self._add_scalar_recursively(writer, o[k], tag, global_step)
        else:
            writer.add_scalar(prefix, o, global_step=global_step)
