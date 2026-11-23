import torch
import numpy as np
from scipy import ndimage
from tqdm import tqdm

import torch

def temporal_distance_to_prediction(valid_pred: torch.Tensor, valid_ref: torch.Tensor, max_distance: float = 15.0):
    """
    valid_pred, valid_ref: (T, N, W, H) binary tensors

    Returns:
        mae: mean absolute error over matched events
        r2:  R² over matched event years
        n_matches: number of matched event pairs
    """
    T, N, W, H = valid_pred.shape
    P = N * W * H

    pred_flat = valid_pred.reshape(T, P)
    ref_flat  = valid_ref.reshape(T, P)

    # Get event indices per pixel
    pred_events = pred_flat.nonzero(as_tuple=False)
    ref_events  = ref_flat.nonzero(as_tuple=False)
    # shape: (num_events, 2)  → [time, pixel]

    # Group by pixel using scatter logic --  Build lists of times per pixel
    pred_times_per_pixel = [[] for _ in range(P)]
    ref_times_per_pixel  = [[] for _ in range(P)]
    for t, p in pred_events:
        pred_times_per_pixel[p.item()].append(t.float())
    for t, p in ref_events:
        ref_times_per_pixel[p.item()].append(t.float())

    pred_all = []
    ref_all  = []
    for p in tqdm(range(P), desc="Calculating temporal distance to prediction over valid pixels", total=P):
        if len(ref_times_per_pixel[p]) == 0:
            continue

        pred_times = torch.tensor(pred_times_per_pixel[p], device=valid_pred.device)
        ref_times  = torch.tensor(ref_times_per_pixel[p],  device=valid_pred.device)

        if len(pred_times) != len(ref_times): #I dont like this
            #simple matching: if different number of events, match as many as possible and ignore the rest
            min_events = min(len(pred_times), len(ref_times))
            pred_times = pred_times[:min_events]
            ref_times = ref_times[:min_events]
        #if len(pred_times) < len(ref_times): #fill with max_distance for unmatched events
        #    pred_times = torch.cat([pred_times, torch.full((len(ref_times) - len(pred_times),), max_distance, device=valid_pred.device)], dim=0)
        #elif len(pred_times) > len(ref_times): #select closest years for each ref event
        #    pred_times_expanded = pred_times.unsqueeze(0) # (1, P)
        #    ref_times_expanded = ref_times.unsqueeze(1)   # (R, 1)
        #    time_diffs = torch.abs(pred_times_expanded - ref_times_expanded) # (R, P)
        #    closest_pred_indices = torch.argmin(time_diffs, dim=1) # (R,)
        #    pred_times = pred_times[closest_pred_indices] # (R,)

        pred_all.append(pred_times)
        ref_all.append(ref_times)

    if len(pred_all) == 0:
        return torch.tensor(float("nan")), torch.tensor(float("nan")), 0

    pred_all = torch.cat(pred_all)
    ref_all  = torch.cat(ref_all)

    differences_ = ref_all - pred_all

    mae = torch.nanmean(torch.abs(differences_))
    r2 = 1 - torch.sum(differences_**2) / torch.sum((ref_all - ref_all.mean())**2)

    return mae.cpu().detach()/max_distance, r2.cpu().detach()


def relative_area_error(valid_pred: torch.Tensor, valid_ref: torch.Tensor, maximum_error_ratio: float = 15.0):
    """
    pred, target: (T, N, W, H) binary tensors
    returns: (T, N,) relative area error
    """
    A_pred = valid_pred.sum(dim=(1, 2, 3)).float() # (T,)
    A_true = valid_ref.sum(dim=(1, 2, 3)).float() # (T,)

    differences_ = (A_true - A_pred)
    ratio_yearly = torch.abs(differences_) / A_true #(A_true.clip(min=eps))
    ratio_yearly[A_true == 0] = torch.nan
    ratio_yearly = ratio_yearly/maximum_error_ratio #codify -- so maximum error is 15 times the true area

    ratio_overall = torch.abs(A_true.sum() - A_pred.sum()) / A_true.sum()
    ratio_overall = ratio_overall/maximum_error_ratio

    #R2
    r2_overall = 1 - torch.sum(differences_**2) / torch.sum((A_true - A_true.mean())**2) 

    return ratio_yearly.cpu().detach(), ratio_overall.cpu().detach(), r2_overall.cpu().detach()


def object_level_recall_4d(valid_pred: torch.Tensor, valid_ref: torch.Tensor, connectivity: int = 1, coverage_threshold=1.0):
    """
    Object-level recall for binary segmentation over (T, N, W, H).

    Args:
        pred: (T, N, W, H) binary tensor
        ref:   (T, N, W, H) binary tensor
        connectivity: 1 = 4-connectivity, 2 = 8-connectivity

    Returns:
        recall: (T, N) tensor with object-level recall per time/sample
    """
    T, N, W, H = valid_pred.shape
    structure = ndimage.generate_binary_structure(2, connectivity)

    recall = torch.zeros((T, N), dtype=torch.float32)
    for t in range(T):
        for n in range(N):
            ref_slice = valid_ref[t, n]
            ref_labeled, num_ref = ndimage.label(ref_slice.cpu().numpy(), structure=structure)
            if num_ref == 0:
                recall[t, n] = torch.nan
                continue

            pred_slice = valid_pred[t, n]            
            ref_labeled = torch.as_tensor(ref_labeled, device=valid_ref.device)
            
            recalled = 0
            for gt_id in range(1, num_ref + 1):
                gt_mask = (ref_labeled == gt_id)
                gt_area = gt_mask.sum()
            
                overlap = pred_slice & gt_mask
                covered_fraction = overlap.sum() / gt_area

                if covered_fraction >= coverage_threshold:
                    recalled += 1

            recall[t, n] = recalled / num_ref
    return recall.cpu().detach()