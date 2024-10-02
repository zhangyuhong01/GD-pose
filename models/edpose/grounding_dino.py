from collections import OrderedDict
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import sys
sys.path.insert(0, '/comp_robot/zhangyuhong1/code2/ED-Pose')
from util.misc import NestedTensor, clean_state_dict, is_main_process
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from gdino_service.models.module import NestedTensor
from gdino_service.worker import Model as GDModel
from omegaconf import OmegaConf
from gdino_service.service_wrapper import ServiceWrapper as GDWorker
#from models.edpose.backbones.prompt_text_dict import PromptTextDict
# -*- coding: utf-8 -*-
import logging
import copy
from collections import defaultdict
from collections import OrderedDict
import torch
from torch import nn
import os
import time
import numpy as np
import pdb
from .position_encoding import build_position_encoding
log = logging.getLogger(os.path.relpath(os.path.abspath(__file__), os.getcwd()))
import math

class PromptTextDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_cat_id = 1
        self.cats = None

        self.encoded_text = None  # (1, token#, fea_dim), tokens=[begin, tokens-for-cat-ids[0], ..., end]
        self.text_self_attention_masks = None  # (1, token#, token#), block-diag matrix
        self.text_token_mask = None  # (1, token#), [True, True, ..., True]
        self.position_ids = None  # (1, token#), [0,0,1,2,0,1,2]

        # the follow tensor is indexed by cat_id. If min_cat_id is 1, index-0 hold the position, but does not respond to any token
        self.cls_pos_map = None  # (cls#, token#) positive map indicating cls-token responding
        self.cls_token_num = None  # (cls#,), token_num of each cls
        self.logit_to_cls_map = None  # (cls# , token#), map logit(query#, token#) to a cls, when post-processing gd returned results

        self.keys = ['encoded_text', 'text_self_attention_masks',
                     'encoded_text', 'text_token_mask', 'position_ids']

    def __getitem__(self, key):
        if key in self.keys:
            return getattr(self, key)
        else:
            raise NotImplementedError(f'unexpected key: {key}')

    def init_value(self, text_dict: dict, cats: dict, trainable=True):
        self.cats = cats
        if cats is not None and min(cats.keys()) == 0:
            self.min_cat_id = 0
        else:
            self.min_cat_id = 1

        self.text_token_mask = text_dict['text_token_mask']  # (1, token#)
        self.text_self_attention_masks = text_dict['text_self_attention_masks']  # (1, token#, token#)
        self.position_ids = text_dict['position_ids']  # (1, token#)
        if trainable:
            self.encoded_text = nn.Parameter(
                text_dict['encoded_text'], requires_grad=True)  # (1, token#, fea_dim=256)
        else:
            self.encoded_text = text_dict['encoded_text']  # tensor, (1, token#, fea_dim=256)

        xs, ys = torch.where(self.position_ids == 0)  # xs is all 0
        ys = ys.tolist()  # [begin, cat_ids[0], ..., cat_ids[N-1], end]

        whole_pos_map = copy.deepcopy(self.text_self_attention_masks)
        for i in range(2, len(ys)):  # even dot(.) should included for text self-att, but it should not respond to any cls
            dot_ind = ys[i] - 1
            whole_pos_map[0, dot_ind] = False
            whole_pos_map[0, :, dot_ind] = False

        if self.min_cat_id == 0:
            pos_map_indices = ys[1:-1]
        else:
            pos_map_indices = ys[:-1]

        self.cls_pos_map = pos_map = whole_pos_map[0, pos_map_indices].to(torch.float)  # (cls#, token#)
        self.cls_token_num = cls_token_num = torch.sum(pos_map, dim=1)  # (cls#,)
        cls_token_num = cls_token_num[:, None]  # (cls#, 1)
        self.logit_to_cls_map = pos_map / cls_token_num
        if self.min_cat_id == 1:  # cat_id=0 should not respond to any token, even though index0 hold the position
            self.logit_to_cls_map[0, 0] = 0.0
            self.cls_pos_map[0, 0] = 0
            self.cls_token_num[0] = 0
        # pdb.set_trace()

    def prepare_data(self, to_detach=True):
        keys = self.keys
        dic = {key: self[key] for key in keys}
        if self.encoded_text != None:
            if to_detach:
                dic['encoded_text'] = self.encoded_text.detach()
            else:
                dic['encoded_text'] = self.encoded_text
        else:
            raise NotImplementedError(f'PromptTextDict must be init before')
        return dic

    def prepare_pos_map_for_anns(self, anns):
        indices = [ann['category_id'] for ann in anns]
        pos_map = self.cls_pos_map[indices]
        return pos_map

    def resolve_cat_id(self, logits, need_sigmoid=True):
        if need_sigmoid:
            logits = logits.sigmoid()  # (bs, q#, token#) to (bs, q#, token#)
        cls_scores = logits @ self.logit_to_cls_map.T  # (bs, q#, token#) @ (token#, cls#) -> (bs, q#, cls#)
        max_scores, cat_ids = torch.max(cls_scores, dim=-1)

        return max_scores, cat_ids, cls_scores

    def get_cats(self, cat_ids):
        if isinstance(cat_ids, torch.Tensor):
            cat_ids = cat_ids.cpu().tolist()

        if isinstance(cat_ids[0], list):
            return [[self.cats[cat_id] for cat_id in bs_cat_ids] for bs_cat_ids in cat_ids]
        else:
            return [self.cats[cat_id] for cat_id in cat_ids]










class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list):
        super().__init__()
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class DINOXFeature(BackboneBase):
# class DINOXFeature(nn.Module):    
    def __init__(self, cat_names, position_embedding=None, cfg=None, init_cfg=None,):
        return_interm_indices = [0,1,2,3]
        #num_channels_all = [256, 512, 1024, 2048]
        num_channels_all = [256, 256, 256, 256]
        num_channels = num_channels_all[4-len(return_interm_indices):]
        train_backbone = False
        backbone = 'dinox'
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)
        #super().__init__()
        if cfg.gd_cfg.mode == 'remote':
            self.gd = GDWorker(**cfg.cfg.gd_cfg)
        elif cfg.gd_cfg.mode == 'local':
            from gdino_service.worker import Model as GDModel
            model = GDModel(**cfg.gd_cfg.model)
            self.gd = GDWorker(mode='local', url=None, model=model)
        else:
            raise NotImplementedError(f'unexpected GD: {cfg}')
        self.accumulative_batching = cfg.get('accumulative_batching', 1)
        self.cat_names = cat_names
        # self.sum_bb_fea = cfg.gd_cfg.get('sum_bb_fea', False)
        # self.cat_prj_bb_fea = cfg.get('cat_prj_bb_fea', False)
        self.memory_key = cfg.get('memory_key', 'enc_fea_map_4')
        self.query_key = cfg.get('query_key', 'dt_fea_0')

        text = '.'.join(self.cat_names) + '.'

        self.text_dict = PromptTextDict()
        tmp = self.gd.init_prompt(text=text)

        cat_dic = {idx: cat_name for idx, cat_name in enumerate(cat_names)}
        self.text_dict.init_value(tmp, trainable=False, cats=cat_dic)
        self.position_embedding = position_embedding    
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)


    def roi_align_forward(self, side_out, x, pred_boxes):
        if self.roi_align is not None and pred_boxes is not None:
            align_box = [b for b in pred_boxes]
            image_size = [tuple(s.shape[1:]) for s in x]
            side_roi_fea = self.roi_align({i: side_out[i] for i in range(len(side_out))}, align_box, image_size).view(
                *list(pred_boxes.shape[:2]), -1)
        else:
            side_roi_fea = None
        return side_roi_fea

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        
        text_dict = self.text_dict.prepare_data()
        B, H, W = x.mask.shape
        # import pdb
        # pdb.set_trace()

        if self.accumulative_batching > 0 and B > self.accumulative_batching:
            it = math.ceil(B / self.accumulative_batching)
            res = []
            for i in range(it):
                begin = i * self.accumulative_batching
                end = begin + self.accumulative_batching
                # if end > B:
                #     end = B
                #     begin = B - self.accumulative_batching_num
                tensors = x.tensors[begin:end]
                mask = x.mask[begin:end]
                sample = NestedTensor(tensors, mask)
                # print_log(f'{sample.shape} {text_dict["encoded_text"].shape=}', logger='current')
                tmp = self.gd.extract_feature_only(samples=sample, text_dict=text_dict, original_wh=None)
                res.append(tmp)

            tmp = {key: torch.cat([o[key] for o in res], dim=0) for key in tmp.keys()}
            # print_log(f'{tmp["enc_fea_map_5"].shape=} {tmp["bb_fea_map_mask_0"].shape=}')
        else:
            tmp = self.gd.extract_feature_only(samples=x, text_dict=text_dict, original_wh=None)
        out = []
        for i in range(4):
            # pdb.set_trace()
            tensors = tmp[f'prj_bb_fea_map_{i}'].permute(0, 3, 1, 2).contiguous()
            mask = tmp[f'bb_fea_map_mask_{i}']
            sample = NestedTensor(tensors, mask)
            out.append(sample)
        # out = [val for key, val in tmp.items() if (key.startswith('prj_bb_fea_map'))]
        # out_mask = [val for key, val in tmp.items() if (key.startswith('bb_fea_map_mask_'))]
        # out = torch.cat(out, dim=1)
        
        # pdb.set_trace()
        # out = torch.cat(list(out), dim=0)
        # out_mask = torch.cat(list(out_mask), dim=0)
        # output = NestedTensor(out, out_mask)
 
        
        # out = {key: val for key, val in tmp.items() if (key.startswith('prj_bb_fea_map'))}
        # pos = []
        # for name, x in enumerate(out):
        #     out.append(x)
        #     pos.append(self.position_embedding(x).to(x.dtype))
        # for key in ['pred_logits', 'pred_boxes_cs', 'pred_boxes']:
        #     out[key] = tmp[key]
        # pdb.set_trace()
        # out['memory'] = tmp[self.memory_key]
        # out['query'] = tmp[self.query_key]
        pos = []
        for name, x in enumerate(out):
            pos.append(self.position_embedding(x).to(x.tensors.dtype))
        
        del tmp  # all cuda tensors, release GPU memory
        # if out['query'].shape[-1] < out['memory'].shape[-1]:
        #     N = out['memory'].shape[-1] // 256
        #     out['query'] = torch.cat([out['query'], ]*N, dim=-1)[..., :out['memory'].shape[-1]]
        # # if self.side_policy is not None:
        # #     out['side_roi_fea'] = side_roi_fea
        # # pdb.set_trace()
        # scores, cat_ids, score_matrix = self.text_dict.resolve_cat_id(logits=out['pred_logits'], need_sigmoid=True)
        # out['pred_scores'] = scores
        # out['pred_cat_ids'] = cat_ids
        # out['pred_score_matrix'] = score_matrix
        
        return out, pos


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_dinox_backbone(args):
    query_cat_num = 2
    memory_key = 'enc_fea_map_4'
    query_key = 'dt_fea_0'
    gd_img_size=1280
    position_embedding = build_position_encoding(args)
    if args.dinox_backbone:
        bb_cfg = dict(
                type='dinox',
                cat_names=['Person',],
                gd_cfg =dict(mode='local', url=None, sum_bb_fea=False,
                                    model=dict(
                                        cfg_path='/comp_robot/shock/share/checkpoints/grounding-dino/configs/demo/gd1.5pro.py',
                                        ckpt_path='/comp_robot/zengzhaoyang/share/gd1.5ckpt/gd1.5pro.pth',
                                        short_edge=gd_img_size,
                                        long_edge=gd_img_size,
                                        padding=True,
                                        remote=False
                                    )),
                accumulative_batching=False,
                accumulative_batching_num=2,
                cat_prj_bb_fea=False,
                cat_bb_fea=True,
                memory_key=memory_key,
                query_key=query_key,
                )

        bb_cfg = OmegaConf.create(bb_cfg)
 
        dinox_backbone = DINOXFeature(bb_cfg['cat_names'], cfg=bb_cfg, position_embedding=position_embedding)
        # model = Joiner(dinox_backbone, position_embedding)
        # return model
    
    return dinox_backbone

if __name__ == "__main__":
    args = OmegaConf.create({})
    args.backbone = 'dinox'
    dionx_backbone = build_backbone(args)

    samples = torch.rand(2, 3, 1280, 1280)
    samples = samples.to('cuda')
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)
    features = dionx_backbone(samples)
    import pdb
    pdb.set_trace()
    print(features.keys())
    
    
      

        


    