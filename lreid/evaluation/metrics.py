import numpy as np
from sklearn import metrics as sk_metrics
from collections import defaultdict

def compute_retrieval_metrics(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Compute retrieval metrics: R@1, R@5, R@10, mAP (macro)"""
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        # Pad cmc to max_rank length
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), mode='constant', constant_values=1)
        else:
            cmc = cmc[:max_rank]
        
        all_cmc.append(cmc)
        num_valid_q += 1
        
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    if num_valid_q == 0:
        return {
            'R@1': 0.0, 'R@5': 0.0, 'R@10': 0.0, 'mAP': 0.0,
            'cmc': np.zeros(max_rank)
        }
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return {
        'R@1': all_cmc[0] * 100,
        'R@5': all_cmc[4] * 100 if max_rank >= 5 else 0.0,
        'R@10': all_cmc[9] * 100 if max_rank >= 10 else 0.0,
        'mAP': mAP * 100,
        'cmc': all_cmc * 100
    }


def compute_map_macro(distmat, q_pids, g_pids, q_camids, g_camids):
    """Compute macro-averaged mAP (per-class average)"""
    unique_pids = np.unique(q_pids)
    aps_per_class = []
    
    for pid in unique_pids:
        q_mask = q_pids == pid
        if not np.any(q_mask):
            continue
        
        q_distmat = distmat[q_mask]
        q_q_pids = q_pids[q_mask]
        q_q_camids = q_camids[q_mask]
        
        result = compute_retrieval_metrics(q_distmat, q_q_pids, g_pids, q_q_camids, g_camids)
        aps_per_class.append(result['mAP'] / 100)
    
    return np.mean(aps_per_class) * 100 if aps_per_class else 0.0


def compute_ood_metrics(scores_seen, scores_unseen):
    """
    Compute OOD detection metrics: AUROC, FPR@TPR95
    scores_seen: confidence scores for seen-class samples (higher = more confident)
    scores_unseen: confidence scores for unseen-class samples
    
    Returns dict with AUROC, FPR95, or None if all classes seen
    """
    if len(scores_unseen) == 0:
        return {'AUROC': None, 'FPR95': None}
    
    y_true = np.concatenate([np.ones(len(scores_seen)), np.zeros(len(scores_unseen))])
    y_scores = np.concatenate([scores_seen, scores_unseen])
    
    auroc = sk_metrics.roc_auc_score(y_true, y_scores)
    
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_scores)
    tpr95_idx = np.where(tpr >= 0.95)[0]
    if len(tpr95_idx) > 0:
        fpr95 = fpr[tpr95_idx[0]]
    else:
        fpr95 = 1.0
    
    return {'AUROC': auroc, 'FPR95': fpr95}


def compute_lifelong_metrics(map_per_task, num_classes_per_task):
    """
    Compute lifelong learning metrics: plasticity, forgetting, overall
    
    Args:
        map_per_task: list of mAP values after each task [task1_map, task2_map, ...]
        num_classes_per_task: list of number of classes per task
    
    Returns:
        dict with plasticity, forgetting, overall
    """
    if len(map_per_task) < 2:
        return {'plasticity': map_per_task[0] if map_per_task else 0.0,
                'forgetting': 0.0,
                'overall': map_per_task[0] if map_per_task else 0.0}
    
    map_per_task = np.array(map_per_task)
    
    plasticity = map_per_task[-1]
    
    forgetting_values = []
    for i in range(len(map_per_task) - 1):
        forgetting = map_per_task[i] - map_per_task[-1]
        forgetting_values.append(max(0, forgetting))
    
    forgetting = np.mean(forgetting_values) if forgetting_values else 0.0
    overall = plasticity - forgetting
    
    return {
        'plasticity': plasticity,
        'forgetting': forgetting,
        'overall': overall
    }


def compute_all_metrics(distmat, q_pids, g_pids, q_camids, g_camids,
                        seen_class_ids=None, unseen_class_ids=None,
                        map_per_task=None, num_classes_per_task=None):
    """Compute all metrics at once"""
    results = {}
    
    retrieval = compute_retrieval_metrics(distmat, q_pids, g_pids, q_camids, g_camids)
    results.update({
        'R@1': retrieval['R@1'],
        'R@5': retrieval['R@5'],
        'R@10': retrieval['R@10'],
        'mAP': retrieval['mAP'],
    })
    
    results['mAP_macro'] = compute_map_macro(distmat, q_pids, g_pids, q_camids, g_camids)
    
    if seen_class_ids is not None and unseen_class_ids is not None:
        q_seen_mask = np.isin(q_pids, seen_class_ids)
        q_unseen_mask = np.isin(q_pids, unseen_class_ids)
        
        if np.any(q_seen_mask) and np.any(q_unseen_mask):
            scores_seen = -np.min(distmat[q_seen_mask], axis=1)
            scores_unseen = -np.min(distmat[q_unseen_mask], axis=1)
            ood = compute_ood_metrics(scores_seen, scores_unseen)
            results['AUROC'] = ood['AUROC']
            results['FPR95'] = ood['FPR95']
        else:
            results['AUROC'] = None
            results['FPR95'] = None
    else:
        results['AUROC'] = None
        results['FPR95'] = None
    
    if map_per_task is not None:
        lifelong = compute_lifelong_metrics(map_per_task, num_classes_per_task)
        results['Plasticity'] = lifelong['plasticity']
        results['Forgetting'] = lifelong['forgetting']
        results['Overall'] = lifelong['overall']
    else:
        results['Plasticity'] = None
        results['Forgetting'] = None
        results['Overall'] = None
    
    return results


class MetricsLogger:
    """Logger for saving results to CSV and JSON after each task"""
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.csv_path = os.path.join(output_path, 'results.csv')
        self.json_path = os.path.join(output_path, 'history.json')
        self.history = []
        self._init_csv()
    
    def _init_csv(self):
        import os
        os.makedirs(self.output_path, exist_ok=True)
        header = 'task,numclass,cnn_top1,nme_top1,R@1,R@5,R@10,mAP,AUROC,FPR95,Plasticity,Forgetting,Overall\n'
        with open(self.csv_path, 'w') as f:
            f.write(header)
    
    def _to_serializable(self, val):
        """Convert numpy types to Python native types for JSON serialization"""
        if val is None:
            return None
        if hasattr(val, 'item'):  # numpy scalar
            return val.item()
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    def log_task(self, task_id, num_classes, cnn_top1, nme_top1, 
                 R1, R5, R10, mAP, AUROC, FPR95, Plasticity, Forgetting, Overall):
        import os
        # Convert all values to serializable
        task_id = self._to_serializable(task_id)
        num_classes = self._to_serializable(num_classes)
        cnn_top1 = self._to_serializable(cnn_top1)
        nme_top1 = self._to_serializable(nme_top1)
        R1 = self._to_serializable(R1)
        R5 = self._to_serializable(R5)
        R10 = self._to_serializable(R10)
        mAP = self._to_serializable(mAP)
        AUROC = self._to_serializable(AUROC)
        FPR95 = self._to_serializable(FPR95)
        Plasticity = self._to_serializable(Plasticity)
        Forgetting = self._to_serializable(Forgetting)
        Overall = self._to_serializable(Overall)
        
        row = f'{task_id},{num_classes},{cnn_top1},{nme_top1},{R1},{R5},{R10},{mAP},{AUROC},{FPR95},{Plasticity},{Forgetting},{Overall}\n'
        with open(self.csv_path, 'a') as f:
            f.write(row)
        
        entry = {
            'task': task_id,
            'numclass': num_classes,
            'cnn_top1': cnn_top1,
            'nme_top1': nme_top1,
            'R@1': R1,
            'R@5': R5,
            'R@10': R10,
            'mAP': mAP,
            'AUROC': AUROC,
            'FPR95': FPR95,
            'Plasticity': Plasticity,
            'Forgetting': Forgetting,
            'Overall': Overall
        }
        self.history.append(entry)
        
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_history(self):
        return self.history


import os
import json