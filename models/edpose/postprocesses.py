import copy
import os
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from torch import Tensor
from pycocotools.coco import COCO
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)



# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""
#     def __init__(self, num_select=100, nms_iou_threshold=-1,num_body_points=17) -> None:
#         super().__init__()
#         self.num_select = num_select
#         self.nms_iou_threshold = nms_iou_threshold
#         self.num_body_points=num_body_points
#     @torch.no_grad()
#     def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
#         num_select = self.num_select
#         out_logits, out_bbox, out_keypoints= outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_keypoints']
#         assert len(out_logits) == len(target_sizes)
#         assert target_sizes.shape[1] == 2
#         prob = out_logits.sigmoid()
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
#         scores = topk_values

#         # bbox
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         labels = topk_indexes % out_logits.shape[2]
#         if not_to_xyxy:
#             boxes = out_bbox
#         else:
#             boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

#         if test:
#             assert not not_to_xyxy
#             boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

#         # from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_sizes.unbind(1)
#         scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#         boxes = boxes * scale_fct[:, None, :]


#         # keypoints
#         topk_keypoints = topk_indexes // out_logits.shape[2]
#         labels = topk_indexes % out_logits.shape[2]
#         keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*3))
#         # import pdb
#         # pdb.set_trace()
#         Z_pred = keypoints[:, :, :(self.num_body_points*2)]
#         V_pred = keypoints[:, :, (self.num_body_points*2):]
#         img_h, img_w = target_sizes.unbind(1)
#         Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)[:, None, :]
#         keypoints_res = torch.zeros_like(keypoints)
#         keypoints_res[..., 0::3] = Z_pred[..., 0::2]
#         keypoints_res[..., 1::3] = Z_pred[..., 1::2]
#         keypoints_res[..., 2::3] = V_pred[..., 0::1]


#         if self.nms_iou_threshold > 0:
#             raise NotImplementedError
#             item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
#             # import ipdb; ipdb.set_trace()
#             results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
#         else:
#             results = [{'scores': s, 'labels': l, 'boxes': b, 'keypoints': k} for s, l, b, k in zip(scores, labels, boxes, keypoints_res)]

#         return results
def get_output_shape(
     oldh: int, oldw: int, short_edge_length: int, max_size: int
):
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1,num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.num_body_points=num_body_points
    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        num_select = self.num_select
        out_logits, out_bbox, out_keypoints= outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_keypoints']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values

        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        # import pdb
        # pdb.set_trace()
        # img_hs = []
        # img_ws = []
        # for h, w in zip(img_h, img_w):
            
        #     img_h, img_w = get_output_shape(img_h, img_w, 1280, 1280)
        #     img_hs.append(1280) img_h
        #     img_ws.append(1280) img_w
        # img_h = torch.tensor(img_hs).to(target_sizes.device)
        # img_w = torch.tensor(img_ws).to(target_sizes.device)
        
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]


        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*3))

        Z_pred = keypoints[:, :, :(self.num_body_points*2)]
        V_pred = keypoints[:, :, (self.num_body_points*2):]
        img_h, img_w = target_sizes.unbind(1)
        Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)[:, None, :]
        # Z_pred = Z_pred * torch.stack([1280, 1280], dim=1).repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]
        
        







        if self.nms_iou_threshold > 0:
            raise NotImplementedError
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import ipdb; ipdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'keypoints': k} for s, l, b, k in zip(scores, labels, boxes, keypoints_res)]

        return results
