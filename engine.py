import math
import os
import sys
from typing import Iterable
import os.path as osp
from util.utils import to_device
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from util import box_ops, keypoint_ops
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    long_edge_size = 1280
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.amp.autocast('cuda', args):
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug: # if debug mode, break the loop test of bath training and testing quickly
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat



@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    iou_types = tuple(k for k in ( 'bbox', 'keypoints'))
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    if args.dataset_file=="coco":
        from datasets.coco_eval import CocoEvaluator
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file=="crowdpose":
        from datasets.crowdpose_eval import CocoEvaluator
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    elif args.dataset_file=="humanart":
        from datasets.humanart_eval import CocoEvaluator
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        #targets = postprocessors['bbox'](targets, orig_target_sizes)
        # import pdb
        # pdb.set_trace()
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    # pdb.set_trace()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            stats['coco_eval_keypoints_detr'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()
    return stats, coco_evaluator

# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
#     try:
#         need_tgt_for_training = args.use_dn
#     except:
#         need_tgt_for_training = False
#     model.eval()
#     criterion.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     if not wo_class_error:
#         metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'
#     iou_types = tuple(k for k in ( 'bbox', 'keypoints'))
#     try:
#         useCats = args.useCats
#     except:
#         useCats = True
#     if not useCats:
#         print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
#     if args.dataset_file=="coco":
#         from datasets.coco_eval import CocoEvaluator
#         coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
#     elif args.dataset_file=="crowdpose":
#         from datasets.crowdpose_eval import CocoEvaluator
#         coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
#     elif args.dataset_file=="humanart":
#         from datasets.humanart_eval import CocoEvaluator
#         coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
#     _cnt = 0
#     coco_results = {}
#     coco_targets = {}
#     coco_results['images'] = []
#     coco_results['annotations'] = []
#     coco_targets['images'] = []
#     coco_targets['annotations'] = []
#     import json
#     categories = json.load(open(f'/comp_robot/zhangyuhong1/code2/ED-Pose/data/coco_dir/annotations/person_keypoints_val2017_shot1.json'))['categories']
#     coco_results['categories'] = categories
#     coco_targets['categories'] = categories
#     for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
#         samples = samples.to(device)
#         targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
#         with torch.cuda.amp.autocast(enabled=args.amp):
#             if need_tgt_for_training:
#                 outputs = model(samples, targets)
#             else:
#                 outputs = model(samples)
        
#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes)
        
#         # import pdb
#         # pdb.set_trace()
#         for target, output in zip(targets, results):
#             image_id = target['image_id'].item()
#             pred_keypoints = output['keypoints'].cpu().numpy()[0]
#             gt_keypoints = target['keypoints'].cpu().numpy()
#             # score_list = []
#             # for i in range(50):
#             #     score_list.append(output['scores'][i].item())
#             # import pdb
#             # pdb.set_trace()
            
#             coco_results["annotations"].append({
#                 "id": image_id, 
#                 "image_id": image_id,
#                 "category_id": 1,  # Assuming single category for simplicity
#                 "keypoints": pred_keypoints.flatten().tolist(),
#                 "score": output['scores'][0].item()
#             })
#             num_keypoints = target['num_keypoints'].cpu().numpy()
#             for keypoint in gt_keypoints:
#                 coco_targets["annotations"].append({
#                     "id": image_id, 
#                     "image_id": image_id,
#                     "category_id": 1,  # Assuming single category for simplicity
#                     "keypoints": keypoint.flatten().tolist(),
#                     "score": 1.0,  # Ground truth keypoints have a score of 1.0
#                     "num_keypoints": num_keypoints.tolist()[0]
#                 })

#         if coco_evaluator is not None:
#             res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#             coco_evaluator.update(res)
        
#         _cnt += 1
#         if args.debug:
#             if _cnt % 15 == 0:
#                 print("BREAK!" * 5)
#                 break
#     import json
#     # Save results and targets to JSON files
#     with open(f'{output_dir}/coco_results.json', 'w') as f:
#         json.dump(coco_results, f, indent=4)
#     with open(f'{output_dir}/coco_targets.json', 'w') as f:
#         json.dump(coco_targets, f, indent=4)

#     # Load results and targets into COCOeval
#     from pycocotools.cocoeval import COCOeval
#     from pycocotools.coco import COCO
#     coco_dt = COCO(f'{output_dir}/coco_results.json')
#     coco_gt = COCO(f'{output_dir}/coco_targets.json')

#     coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#     import pdb
#     pdb.set_trace()
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#     # pdb.set_trace()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#             stats['coco_eval_keypoints_detr'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()
#     return stats, coco_evaluator

@torch.no_grad()
def inference_vis(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # for visualize
    from util.visualizer import COCOVisualizer
    from pycocotools.coco import COCO

    COCO_PATH = os.environ.get("EDPOSE_COCO_PATH")
    cocodir = COCO_PATH + '/annotations/person_keypoints_val2017.json'
    coco = COCO(cocodir)
    vslzr = COCOVisualizer(coco)
    _cnt = 0
    # import pdb
    # pdb.set_trace() 
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.stack([torch.tensor([1280,1280], device="cuda:0") for t in targets], dim=0)
        images = samples.tensors.detach().cpu()

        #output_viss = postprocessors['bbox'](outputs, torch.ones_like(orig_target_sizes))
        output_viss = postprocessors['bbox'](outputs, torch.ones_like(orig_target_sizes))
        # import pdb
        # pdb.set_trace()
        thersholds = [0.1, 0.13, 0.3, 0.5] # set a thershold
        for _idx, (tgt, output_vis) in enumerate(zip(targets, output_viss)):
            image_id=tgt["image_id"]
            scores = output_vis['scores']
            boxes = box_ops.box_xyxy_to_cxcywh(output_vis['boxes'])
            keypoints = output_vis['keypoints']
            keypoints = keypoint_ops.keypoint_xyzxyz_to_xyxyzz(keypoints)
            for thershold in thersholds:
                select_mask = scores > thershold
                pred_dict = {
                    'boxes': boxes[select_mask],
                    'size': tgt['size'],
                    'image_id': tgt['image_id'],
                    'keypoints': keypoints[select_mask],
                }
                vslzr.visualize(images[_idx], pred_dict, caption=f"{int(image_id)}", savedir=os.path.join(args.output_dir, 'vis'))
    return





