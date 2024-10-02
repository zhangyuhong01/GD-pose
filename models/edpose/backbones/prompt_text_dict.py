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
log = logging.getLogger(os.path.relpath(os.path.abspath(__file__), os.getcwd()))


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
