import torch
import numpy as np
from lreid.tools import time_now, CatMeter
from lreid.evaluation import (fast_evaluate_rank, compute_distance_matrix, compute_ood_metrics)


def _get_seen_global_pids(loaders, current_step):
    """Get seen global PIDs (0-24) up to current step for IP102"""
    if hasattr(loaders, 'ip102_task_splits'):
        # Build mapping from original class ID to global PID
        orig_to_global = {}
        global_pid = 0
        for task_classes in loaders.ip102_task_splits:
            for orig_c in task_classes:
                orig_to_global[orig_c] = global_pid
                global_pid += 1
        
        seen = set()
        for i in range(current_step + 1):
            for orig_c in loaders.ip102_task_splits[i]:
                seen.add(orig_to_global[orig_c])
        return seen
    return None


def fast_test_p_s(config, base, loaders, current_step, if_test_forget=True):
    """
    Open-world evaluation for IP102:
    - Test on FULL 25 classes
    - Separate SEEN (trained) vs UNSEEN (not yet trained) classes
    - Metrics:
        * R@1, mAP on SEEN classes (closed-world)
        * AUROC, FPR@TPR95 for open-set (seen vs unseen)
    """
    base.set_all_model_eval()
    print(f'****** start perform fast testing! (step {current_step}) ******')

    def _cmc_map(_query_features_meter, _gallery_features_meter, _query_pids_meter, _gallery_pids_meter,
                  _query_cids_meter, _gallery_cids_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()

        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      _query_pids_meter.get_val_numpy(),
                                      _gallery_pids_meter.get_val_numpy(),
                                      _query_cids_meter.get_val_numpy(),
                                      _gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=True)
        return CMC[0] * 100, mAP * 100

    def _compute_open_set_metrics(query_features, gallery_features, query_pids, gallery_pids, seen_global_pids):
        """Compute open-set metrics: AUROC, FPR@TPR95 for seen vs unseen"""
        # For each query, get min distance to gallery
        distance_matrix = compute_distance_matrix(query_features, gallery_features, config.test_metric)
        distance_matrix = distance_matrix.data.cpu().numpy()
        
        # Min distance per query (lower = more confident match)
        min_dist = np.min(distance_matrix, axis=1)
        scores = -min_dist  # higher = more confident
        
        query_pids_np = query_pids
        seen_mask = np.isin(query_pids_np, list(seen_global_pids))
        unseen_mask = ~seen_mask
        
        if np.sum(seen_mask) == 0 or np.sum(unseen_mask) == 0:
            return None, None
        
        scores_seen = scores[seen_mask]
        scores_unseen = scores[unseen_mask]
        
        ood = compute_ood_metrics(scores_seen, scores_unseen)
        return ood['AUROC'], ood['FPR95']

    results_dict = {}
    seen_global_pids = _get_seen_global_pids(loaders, current_step)
    print(f'  Step {current_step}: Seen global PIDs = {sorted(seen_global_pids) if seen_global_pids else "None"}')

    for dataset_name, temp_loaders in loaders.test_loader_dict.items():
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
        query_fuse_features_meter, query_fuse_pids_meter, query_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_fuse_features_meter, gallery_fuse_pids_meter, gallery_fuse_cids_meter = CatMeter(), CatMeter(), CatMeter()

        print(time_now(), f' {dataset_name} feature start ')
        with torch.no_grad():
            for loader_id, loader in enumerate(temp_loaders):
                for data in loader:
                    images, pids, cids = data[0:3]
                    images = images.to(base.device)
                    features, featuremaps = base.model_dict['tasknet'](images, current_step)
                    
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)

        print(time_now(), f' {dataset_name} feature done')

        # === CLOSED-WORLD: SEEN classes only ===
        query_pids = query_pids_meter.get_val_numpy()
        gallery_pids = gallery_pids_meter.get_val_numpy()
        query_cids = query_cids_meter.get_val_numpy()
        gallery_cids = gallery_cids_meter.get_val_numpy()
        query_features = query_features_meter.get_val()
        gallery_features = gallery_features_meter.get_val()

        if seen_global_pids is not None:
            # Filter to SEEN classes for closed-world metrics
            seen_mask_q = np.isin(query_pids, list(seen_global_pids))
            seen_mask_g = np.isin(gallery_pids, list(seen_global_pids))
            
            if np.sum(seen_mask_q) > 0 and np.sum(seen_mask_g) > 0:
                q_feat_seen = query_features[seen_mask_q]
                g_feat_seen = gallery_features[seen_mask_g]
                q_pids_seen = query_pids[seen_mask_q]
                g_pids_seen = gallery_pids[seen_mask_g]
                q_cids_seen = query_cids[seen_mask_q]
                g_cids_seen = gallery_cids[seen_mask_g]

                # Closed-world metrics on SEEN only
                dist_seen = compute_distance_matrix(q_feat_seen, g_feat_seen, config.test_metric)
                dist_seen = dist_seen.data.cpu().numpy()
                CMC, mAP = fast_evaluate_rank(dist_seen, q_pids_seen, g_pids_seen, q_cids_seen, g_cids_seen,
                                              max_rank=50, use_metric_cuhk03=False, use_cython=True)
                results_dict[f'{dataset_name}_seen_R@1'] = CMC[0] * 100
                results_dict[f'{dataset_name}_seen_mAP'] = mAP * 100
                print(f'  SEEN ({len(seen_global_pids)} classes): R@1={CMC[0]*100:.2f}%, mAP={mAP*100:.2f}%')

                # === OPEN-WORLD: SEEN vs UNSEEN ===
                auroc, fpr95 = _compute_open_set_metrics(
                    query_features, gallery_features, query_pids, gallery_pids, seen_global_pids)
                if auroc is not None:
                    results_dict[f'{dataset_name}_AUROC'] = auroc * 100
                    results_dict[f'{dataset_name}_FPR95'] = fpr95 * 100
                    print(f'  OPEN-SET: AUROC={auroc*100:.2f}%, FPR@TPR95={fpr95*100:.2f}%')
            else:
                results_dict[f'{dataset_name}_seen_R@1'] = 0.0
                results_dict[f'{dataset_name}_seen_mAP'] = 0.0
                print(f'  SEEN: No valid query/gallery pairs')
        else:
            # Fallback: evaluate on all (for non-IP102 datasets)
            dist = compute_distance_matrix(query_features, gallery_features, config.test_metric)
            dist = dist.data.cpu().numpy()
            CMC, mAP = fast_evaluate_rank(dist, query_pids, gallery_pids, query_cids, gallery_cids,
                                          max_rank=50, use_metric_cuhk03=False, use_cython=True)
            results_dict[f'{dataset_name}_R@1'] = CMC[0] * 100
            results_dict[f'{dataset_name}_mAP'] = mAP * 100

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    return results_dict, results_str


def save_and_fast_test_p_s(config, base, loaders, current_step, current_epoch, if_test_forget=True):
    """Same as fast_test_p_s but with epoch logging"""
    return fast_test_p_s(config, base, loaders, current_step, if_test_forget)