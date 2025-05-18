"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot_val

from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk
import dataloaders.augutils as myaug

from util.metric import Metric
from util.consts import IMG_SIZE
from util.utils import cca, sliding_window_confidence_segmentation, plot_3d_bar_probabilities, save_pred_gt_fig, plot_heatmap_of_probs
from config_ssl_upload import ex

from tqdm import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import pandas as pd
import cv2
from collections import defaultdict

from util.utils import set_seed, t2n, to01, compose_wt_simple
# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"


def test_time_training(_config, model, image, prediction):
    model.train()
    data_name = _config['dataset']
    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config['ignore_label'], weight=my_weight)
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=_config['lr'], eps=1e-5)
    else:
        raise NotImplementedError
    optimizer.zero_grad()
    scheduler = MultiStepLR(
        optimizer, milestones=_config['lr_milestones'],  gamma=_config['lr_step_gamma'])
    
    tr_transforms = myaug.transform_with_label(
        {'aug': myaug.get_aug(_config['which_aug'], _config['input_size'][0])})
   
    comp = np.concatenate([image.transpose(1, 2, 0), prediction[None,...].transpose(1,2,0)], axis= -1)
    print("Test Time Training...")
    pbar = tqdm(range(_config['n_steps']))
    for idx in pbar:
        query_image, query_label = tr_transforms(comp, c_img=image.shape[0], c_label=1, nclass=2, use_onehot=False)
        support_image, support_label = tr_transforms(comp, c_img=image.shape[0], c_label=1, nclass=2, use_onehot=False)
        query_label = torch.from_numpy(query_label.transpose(2,1,0)).cuda().long()

        query_images = [torch.from_numpy(query_image.transpose(2, 1, 0)).unsqueeze(0).cuda().float().requires_grad_(True)]
        support_fg_mask = [[torch.from_numpy(support_label.transpose(2, 1, 0)).cuda().float().requires_grad_(True)]]
        support_bg_mask = [[torch.from_numpy(1 - support_label.transpose(2, 1, 0)).cuda().float().requires_grad_(True)]]
        support_images = [[torch.from_numpy(support_image.transpose(2, 1, 0)).unsqueeze(0).cuda().float().requires_grad_(True)]]
    
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(query_images[0][0,0].cpu().numpy())
        # ax[1].imshow(support_image[...,0])
        # ax[1].imshow(support_label[...,0], alpha=0.5)
        # fig.savefig("debug/query_support_ttt.png") 
        out = model(support_images, support_fg_mask, support_bg_mask, query_images, isval=False, val_wsize=None)
        query_pred, align_loss, _, _, _, _, _ = out
        # fig, ax = plt.subplots(1, 2)
        # pred = np.array(query_pred.argmax(dim=1)[0].cpu())
        # ax[0].imshow(query_images[0][0,0].cpu().numpy())
        # ax[0].imshow(pred, alpha=0.5)
        # ax[1].imshow(support_image[...,0])
        # ax[1].imshow(support_label[...,0], alpha=0.5)
        # fig.savefig("debug/ttt.png")
        loss = 0.0
        loss += criterion(query_pred.float(), query_label.long())
        loss += align_loss
        loss.backward()
    
        if (idx + 1) % _config['grad_accumulation_steps'] == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    model.eval()
    return model


def plot_pred_gt_support(query_image, pred, gt, support_images, support_masks, score=None, save_path="debug/pred_vs_gt"):
    """
    Save 5 key images: support images, support mask, query, ground truth and prediction.
    Handles both grayscale and RGB images consistently with the same mask color.
    
    Args:
        query_image: Query image tensor (grayscale or RGB)
        pred: 2d tensor where 1 represents foreground and 0 represents background
        gt: 2d tensor where 1 represents foreground and 0 represents background
        support_images: Support image tensors (grayscale or RGB)
        support_masks: Support mask tensors
        score: Optional score to add to filename
        save_path: Base path without extension for saving images
    """
    # Create directory for this case
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Process query image - ensure HxWxC format for visualization
    query_image = query_image.clone().detach().cpu()
    if len(query_image.shape) == 3 and query_image.shape[0] <= 3:  # CHW format
        query_image = query_image.permute(1, 2, 0)
    
    # Handle grayscale vs RGB consistently
    if len(query_image.shape) == 2 or (len(query_image.shape) == 3 and query_image.shape[2] == 1):
        # For grayscale, use cmap='gray' for visualization
        is_grayscale = True
        if len(query_image.shape) == 3:
            query_image = query_image.squeeze(2)  # Remove channel dimension for grayscale
    else:
        is_grayscale = False
    
    # Normalize image for visualization
    query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min() + 1e-8)
    
    # Convert pred and gt to numpy for visualization
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    
    # Create colormap for mask overlays - using a consistent red colormap
    mask_cmap = plt.cm.get_cmap('YlOrRd')  # Yellow-Orange-Red colormap

    # Generate color masks with alpha values
    pred_rgba = mask_cmap(pred_np)
    pred_rgba[..., 3] = pred_np * 0.7  # Last channel is alpha - semitransparent where mask=1
    
    gt_rgba = mask_cmap(gt_np)
    gt_rgba[..., 3] = gt_np * 0.7  # Last channel is alpha - semitransparent where mask=1
    
    # 1. Save query image (original)
    plt.figure(figsize=(10, 10))
    if is_grayscale:
        plt.imshow(query_image, cmap='gray')
    else:
        plt.imshow(query_image)
    plt.axis('off')
    # Remove padding/whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"{save_path}/query.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 2. Save query image with prediction overlay
    plt.figure(figsize=(10, 10))
    if is_grayscale:
        plt.imshow(query_image, cmap='gray')
    else:
        plt.imshow(query_image)
    plt.imshow(pred_rgba)
    plt.axis('off')
    # Remove padding/whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"{save_path}/pred.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 3. Save query image with ground truth overlay
    plt.figure(figsize=(10, 10))
    if is_grayscale:
        plt.imshow(query_image, cmap='gray')
    else:
        plt.imshow(query_image)
    plt.imshow(gt_rgba)
    plt.axis('off')
    # Remove padding/whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(f"{save_path}/gt.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Process and save support images and masks (just the first one for brevity)
    if support_images is not None:
        if isinstance(support_images, list):
            support_images = torch.cat([tensor for tensor_list in support_images for tensor in tensor_list], dim=0).clone().detach()
        if isinstance(support_masks, list):
            support_masks = torch.cat([tensor for tensor_list in support_masks for tensor in tensor_list], dim=0).clone().detach()
        
        # Move to CPU for processing
        support_images = support_images.cpu()
        support_masks = support_masks.cpu()
        
        # Handle different dimensions of support images
        if len(support_images.shape) == 4:  # NCHW format
            # Convert to NHWC for visualization
            support_images = support_images.permute(0, 2, 3, 1)
        
        # Just process the first support image
        i = 0
        if support_images.shape[0] > 0:
            support_img = support_images[i].clone()
            support_mask = support_masks[i].clone()
            
            # Check if grayscale or RGB
            if support_img.shape[-1] == 1:  # Last dimension is channels
                support_img = support_img.squeeze(-1)  # Remove channel dimension
                support_is_gray = True
            elif support_img.shape[-1] == 3:
                support_is_gray = False
            else:  # Assume it's grayscale if not 1 or 3 channels
                support_is_gray = True
            
            # Normalize support image
            support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min() + 1e-8)
            
            # 4. Save support image only
            plt.figure(figsize=(10, 10))
            if support_is_gray:
                plt.imshow(support_img, cmap='gray')
            else:
                plt.imshow(support_img)
            plt.axis('off')
            # Remove padding/whitespace
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(f"{save_path}/support_1.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 5. Save support mask only (direct mask visualization similar to gt/pred)
            plt.figure(figsize=(10, 10))
            
            # Process support mask exactly like gt/pred
            support_mask_np = support_mask.cpu().numpy()
            support_mask_rgba = mask_cmap(support_mask_np)
            support_mask_rgba[..., 3] = support_mask_np * 0.7  # Last channel is alpha - semitransparent where mask=1
            
            if is_grayscale:
                plt.imshow(support_img, cmap='gray')
            else:
                plt.imshow(support_img)
            plt.imshow(support_mask_rgba)
            plt.axis('off')
            # Remove padding/whitespace
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(f"{save_path}/support_mask.png", bbox_inches='tight', pad_inches=0)
            plt.close()

def get_dice_iou_precision_recall(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    """
    if gt.sum() == 0:
        print("gt is all background")
        return {"dice": 0, "precision": 0, "recall": 0, "iou": 0}

    # Resize pred to match gt dimensions if they're different
    if pred.shape != gt.shape:
        print(f"Resizing prediction from {pred.shape} to match ground truth {gt.shape}")
        # Use interpolate to resize pred to match gt dimensions
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0).float(), 
            size=gt.shape, 
            mode='nearest'
        ).squeeze(0).squeeze(0)

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/bad_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = FewShotSeg(image_size=_config['input_size'][0],
                           pretrained_path=_config['reload_model_path'], cfg=_config['model'])

    model = model.cuda()
    model.eval()

    _log.info('###### Load data ######')
    # Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix' or data_name == 'SABS_Superpix_448' or data_name == 'SABS_Superpix_672':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2_Superpix_672':
        baseset_name = 'CHAOST2'
        max_label = 4
    elif 'lits' in data_name.lower():
        baseset_name = 'LITS17'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - \
        DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]


    _log.info(
        f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(
        f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'SABS':
        tr_parent = SuperpixelDataset(  # base dataset
            which_dataset=baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split=_config['eval_fold'],
            mode='val',  # 'train',
            # dummy entry for superpixel dataset
            min_fg=str(_config["min_fg_data"]),
            image_size=_config['input_size'][0],
            transforms=None,
            nsup=_config['task']['n_shots'],
            scan_per_load=_config['scan_per_load'],
            exclude_list=_config["exclude_cls_list"],
            superpix_scale=_config["superpix_scale"],
            fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
                data_name == 'CHAOST2_Superpix') else None,
            use_clahe=_config['use_clahe'],
            norm_mean=0.18792 * 256 if baseset_name == 'LITS17' else None,
            norm_std=0.25886 * 256 if baseset_name == 'LITS17' else None
        )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality='MR', fids=None)

    te_dataset, te_parent = med_fewshot_val(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split=_config['eval_fold'],
        scan_per_load=_config['scan_per_load'],
        act_labels=test_labels,
        npart=_config['task']['npart'],
        nsup=_config['task']['n_shots'],
        extern_normalize_func=norm_func,
        image_size=_config["input_size"][0],
        use_clahe=_config['use_clahe'],
        use_3_slices=_config["use_3_slices"]
    )

    # dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Set validation nodes ######')
    mar_val_metric_node = Metric(max_label=max_label, n_scans=len(
        te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])

    _log.info('###### Starting validation ######')
    mar_val_metric_node.reset()
    if _config["sliding_window_confidence_segmentation"]:
        print("Using sliding window confidence segmentation")  # TODO delete this

    save_pred_buffer = {}  # indexed by class
    
    # For tracking metrics by scan/case
    mean_dice_by_scan = defaultdict(list)
    mean_iou_by_scan = defaultdict(list)
    mean_dice = []
    mean_prec = []
    mean_rec = []
    mean_iou = []

    for curr_lb in test_labels:
        te_dataset.set_curr_cls(curr_lb)
        support_batched = te_parent.get_support(curr_class=curr_lb, class_idx=[
                                                curr_lb], scan_idx=_config["support_idx"], npart=_config['task']['npart'])

        # way(1 for now) x part x shot x 3 x H x W] #
        support_images = [[shot.cuda() for shot in way]
                            for way in support_batched['support_images']]  # way x part x [shot x C x H x W]
        suffix = 'mask'
        support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                            for way in support_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                            for way in support_batched['support_mask']]

        curr_scan_count = -1  # counting for current scan
        _lb_buffer = {}  # indexed by scan
        _lb_vis_buffer = {}

        last_qpart = 0  # used as indicator for adding result to buffer

        for idx, sample_batched in enumerate(tqdm(testloader)):
            # we assume batch size for query is 1
            _scan_id = sample_batched["scan_id"][0]
            if _scan_id in te_parent.potential_support_sid:  # skip the support scan, don't include that to query
                continue
            if sample_batched["is_start"]:
                ii = 0
                curr_scan_count += 1
                print(
                    f"Processing scan {curr_scan_count + 1} / {len(te_dataset.dataset.pid_curr_load)}")
                _scan_id = sample_batched["scan_id"][0]
                outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                # original image read by itk: Z, H, W, in prediction we use H, W, Z
                outsize = (_config['input_size'][0],
                            _config['input_size'][1], outsize[0])
                _pred = np.zeros(outsize)
                _pred.fill(np.nan)
                # assign proto shows in the query image which proto is assigned to each pixel, proto_grid is the ids of the prototypes in the support image used, support_images are the 3 support images, support_img_parts are the parts of the support images used for each query image
                _vis = {'assigned_proto': [None] * _pred.shape[-1], 'proto_grid': [None] * _pred.shape[-1],
                        'support_images': support_images, 'support_img_parts': [None] * _pred.shape[-1]}

            # the chunck of query, for assignment with support
            q_part = sample_batched["part_assign"]
            query_images = [sample_batched['image'].cuda()]
            query_labels = torch.cat(
                [sample_batched['label'].cuda()], dim=0)
            if 1 not in query_labels and not sample_batched["is_end"] and _config["skip_no_organ_slices"]:
                ii += 1
                continue
            # [way, [part, [shot x C x H x W]]] ->
            # way(1) x shot x [B(1) x C x H x W]
            sup_img_part = [[shot_tensor.unsqueeze(
                0) for shot_tensor in support_images[0][q_part]]]
            sup_fgm_part = [[shot_tensor.unsqueeze(
                0) for shot_tensor in support_fg_mask[0][q_part]]]
            sup_bgm_part = [[shot_tensor.unsqueeze(
                0) for shot_tensor in support_bg_mask[0][q_part]]]

            # query_pred_logits, _, _, assign_mats, proto_grid, _, _ = model(
            #     sup_img_part, sup_fgm_part, sup_bgm_part, query_images, isval=True, val_wsize=_config["val_wsize"], show_viz=True)
            with torch.no_grad():
                out = model(sup_img_part, sup_fgm_part, sup_bgm_part,
                        query_images, isval=True, val_wsize=_config["val_wsize"])
            query_pred_logits, _, _, assign_mats, proto_grid, _, _ = out
            pred = np.array(query_pred_logits.argmax(dim=1)[0].cpu())
                
            if _config["ttt"]: 
                state_dict = model.state_dict()
                model = test_time_training(_config, model, sample_batched['image'].numpy()[0], pred)
                out = model(sup_img_part, sup_fgm_part, sup_bgm_part,
                        query_images, isval=True, val_wsize=_config["val_wsize"])
                query_pred_logits, _, _, assign_mats, proto_grid, _, _ = out
                pred = np.array(query_pred_logits.argmax(dim=1)[0].cpu())
                if _config["reset_after_slice"]:
                    model.load_state_dict(state_dict)
                    
            query_pred = query_pred_logits.argmax(dim=1).cpu()
            query_pred_orig = query_pred.clone()
            query_pred = F.interpolate(query_pred.unsqueeze(
                0).float(), size=query_labels.shape[-2:], mode='nearest').squeeze(0).long().numpy()[0]

            if _config["debug"]:
                save_path = f'debug/preds/scan_{_scan_id}_label_{curr_lb}_{idx}'
                os.makedirs(save_path, exist_ok=True)
                plot_pred_gt_support(
                    query_images[0], 
                    query_pred_orig, 
                    query_labels[0].cpu(),
                    sup_img_part, 
                    sup_fgm_part,
                    save_path=save_path
                )
                
            if _config['do_cca']:
                query_pred = cca(query_pred, query_pred_logits)
                if _config["debug"]:
                    save_path = f'debug/preds/scan_{_scan_id}_label_{curr_lb}_{idx}_after_cca'
                    os.makedirs(save_path, exist_ok=True)
                    plot_pred_gt_support(
                        query_images[0], 
                        torch.from_numpy(query_pred), 
                        query_labels[0].cpu(),
                        sup_img_part, 
                        sup_fgm_part,
                        save_path=save_path
                    )

            _pred[..., ii] = query_pred.copy()

            # Calculate metrics for this slice
            if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']) and not sample_batched["is_end"]:
                mar_val_metric_node.record(query_pred, np.array(
                    query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count)
                
                # Calculate slice-wise metrics
                metrics = get_dice_iou_precision_recall(
                    torch.from_numpy(query_pred), 
                    query_labels[0].cpu()
                )
                mean_dice.append(metrics["dice"])
                mean_prec.append(metrics["precision"])
                mean_rec.append(metrics["recall"])
                mean_iou.append(metrics["iou"])
                
                # Store metrics by scan
                mean_dice_by_scan[_scan_id].append(metrics["dice"])
                mean_iou_by_scan[_scan_id].append(metrics["iou"])
                
                # Save bad predictions
                if metrics["dice"] < 0.6 and _config["debug"]:
                    path = f'debug/bad_preds/scan_{_scan_id}_label_{curr_lb}_{idx}_dice_{metrics["dice"]:.4f}'
                    os.makedirs(path, exist_ok=True)
                    print(f"saving bad prediction to {path}")
                    plot_pred_gt_support(
                        query_images[0], 
                        torch.from_numpy(query_pred), 
                        query_labels[0].cpu(),
                        sup_img_part, 
                        sup_fgm_part,
                        save_path=path
                    )
                
                # Save bad predictions to the run directory
                if metrics["dice"] < 0.6:
                    path = f'{_run.observers[0].dir}/bad_preds/scan_{_scan_id}_label_{curr_lb}_{idx}_dice_{metrics["dice"]:.4f}'
                    os.makedirs(path, exist_ok=True)
                    plot_pred_gt_support(
                        query_images[0], 
                        torch.from_numpy(query_pred), 
                        query_labels[0].cpu(),
                        sup_img_part, 
                        sup_fgm_part,
                        save_path=path
                    )

            ii += 1
            # now check data format
            if sample_batched["is_end"]:
                if _config['dataset'] != 'C0':
                    _lb_buffer[_scan_id] = _pred.transpose(
                        2, 0, 1)  # H, W, Z -> to Z H W
                else:
                    _lb_buffer[_scan_id] = _pred
                # _lb_vis_buffer[_scan_id] = _vis

        save_pred_buffer[str(curr_lb)] = _lb_buffer

        # save results
        for curr_lb, _preds in save_pred_buffer.items():
            for _scan_id, _pred in _preds.items():
                _pred *= float(curr_lb)
                itk_pred = convert_to_sitk(
                    _pred, te_dataset.dataset.info_by_scan[_scan_id])
                fid = os.path.join(
                    f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
                sitk.WriteImage(itk_pred, fid, True)
                _log.info(f'###### {fid} has been saved ######')


    # compute dice scores by scan
    m_classDice, _, m_meanDice, _, m_rawDice = mar_val_metric_node.get_mDice(
        labels=sorted(test_labels), n_scan=None, give_raw=True)

    m_classPrec, _, m_meanPrec, _,  m_classRec, _, m_meanRec, _, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(
        labels=sorted(test_labels), n_scan=None, give_raw=True)

    mar_val_metric_node.reset()  # reset this calculation node

    # Log metrics by scan
    for _scan_id in mean_dice_by_scan.keys():
        _run.log_scalar(f'mar_val_batches_meanDice_{_scan_id}', np.mean(mean_dice_by_scan[_scan_id]))
        _run.log_scalar(f'mar_val_batches_meanIOU_{_scan_id}', np.mean(mean_iou_by_scan[_scan_id]))
        _log.info(f'mar_val batches meanDice_{_scan_id}: {np.mean(mean_dice_by_scan[_scan_id])}')
        _log.info(f'mar_val batches meanIOU_{_scan_id}: {np.mean(mean_iou_by_scan[_scan_id])}')

    # write validation result to log file
    _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
    _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
    _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

    _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
    _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

    _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
    _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

    # Log average metrics as well
    if len(mean_dice) > 0:
        _run.log_scalar('overall_mean_dice', float(np.mean(mean_dice)))
        _run.log_scalar('overall_mean_prec', float(np.mean(mean_prec)))
        _run.log_scalar('overall_mean_rec', float(np.mean(mean_rec)))
        _run.log_scalar('overall_mean_iou', float(np.mean(mean_iou)))

    _log.info(f'mar_val batches classDice: {m_classDice}')
    _log.info(f'mar_val batches meanDice: {m_meanDice}')
    
    _log.info(f'overall_mean_dice: {np.mean(mean_dice)}')
    _log.info(f'overall_mean_iou: {np.mean(mean_iou)}')

    _log.info(f'mar_val batches classPrec: {m_classPrec}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

    _log.info(f'mar_val batches classRec: {m_classRec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')

    print("============ ============")

    _log.info(f'End of validation')
    return 1
