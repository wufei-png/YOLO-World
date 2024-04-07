from mmyolo.registry import MODELS, TASK_UTILS

from typing import List, Optional

import torch
from mmengine.config import ConfigDict
from torch import Tensor
import sys
sys.path.append('../')
from nms.ort_nms import onnx_nms

def pred_by_feat(
                cls_scores: List[Tensor],
                bbox_preds: List[Tensor],
                objectnesses: Optional[List[Tensor]] = None,
                coeff_preds: Optional[List[Tensor]] = None,
                proto_preds: Optional[List[Tensor]] = None,                     
                  **kwargs):
    assert len(cls_scores) == len(bbox_preds)

    bbox_coder_config = ConfigDict(type='DistancePointBBoxCoder')
    prior_generator_config=ConfigDict(
        type='mmdet.MlvlPointGenerator',
        offset=0.5,
        strides=[8, 16, 32])
    
    bbox_coder = TASK_UTILS.build(bbox_coder_config)
    prior_generator = TASK_UTILS.build(prior_generator_config)
    bbox_decoder = bbox_coder.decode
    prior_generate = prior_generator.grid_priors

    dtype = cls_scores[0].dtype
    device = cls_scores[0].device

    nms_func = onnx_nms
    num_base_priors=1
    featmap_strides=[8, 16, 32]
    num_classes=80

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    mlvl_priors = prior_generate(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * num_base_priors, ),
            stride) for featmap_size, stride in zip(
                featmap_sizes, featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    # using score.shape
    text_len = cls_scores[0].shape[1]
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, text_len)
        for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores

    bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                          flatten_stride)
    # 100 0.65 0.25 1000 100
    keep_top_k=100
    iou_threshold=0.65
    score_threshold=0.25
    pre_top_k=1000

    return nms_func(bboxes, scores, keep_top_k, iou_threshold,
                    score_threshold, pre_top_k, keep_top_k)