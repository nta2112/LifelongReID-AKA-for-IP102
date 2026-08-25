import numpy as np
import sys
import os

# Add the lreid evaluation module path directly
sys.path.insert(0, r'D:\Sau_Benh_object\retrieval-img\LifelongReID')

# Import metrics directly without importing the whole lreid package
import importlib.util
spec = importlib.util.spec_from_file_location(
    "metrics", 
    r"D:\Sau_Benh_object\retrieval-img\LifelongReID\lreid\evaluation\metrics.py"
)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)

compute_retrieval_metrics = metrics_module.compute_retrieval_metrics
compute_map_macro = metrics_module.compute_map_macro
compute_ood_metrics = metrics_module.compute_ood_metrics
compute_lifelong_metrics = metrics_module.compute_lifelong_metrics
compute_all_metrics = metrics_module.compute_all_metrics

def test_retrieval_metrics_perfect():
    """Test with perfect retrieval (R@1=1.0, mAP=1.0)"""
    # Create perfect match scenario
    num_q, num_g = 10, 10
    q_pids = np.arange(num_q)
    g_pids = np.arange(num_g)
    q_camids = np.zeros(num_q)
    g_camids = np.ones(num_g)  # Different cameras
    
    # Perfect distance matrix (diagonal = 0, others = 1)
    distmat = np.eye(num_q, num_g) * 0 + (1 - np.eye(num_q, num_g)) * 1
    
    results = compute_retrieval_metrics(distmat, q_pids, g_pids, q_camids, g_camids)
    
    assert abs(results['R@1'] - 100.0) < 0.01, f"R@1 should be 100, got {results['R@1']}"
    assert abs(results['R@5'] - 100.0) < 0.01, f"R@5 should be 100, got {results['R@5']}"
    assert abs(results['R@10'] - 100.0) < 0.01, f"R@10 should be 100, got {results['R@10']}"
    assert abs(results['mAP'] - 100.0) < 0.01, f"mAP should be 100, got {results['mAP']}"
    print("PASS: Perfect retrieval test passed")

def test_retrieval_metrics_random():
    """Test with random retrieval"""
    num_q, num_g = 20, 50
    q_pids = np.random.randint(0, 10, num_q)
    g_pids = np.random.randint(0, 10, num_g)
    q_camids = np.random.randint(0, 2, num_q)
    g_camids = np.random.randint(0, 2, num_g)
    
    distmat = np.random.rand(num_q, num_g)
    
    results = compute_retrieval_metrics(distmat, q_pids, g_pids, q_camids, g_camids)
    
    assert 0 <= results['R@1'] <= 100
    assert 0 <= results['R@5'] <= 100
    assert 0 <= results['R@10'] <= 100
    assert 0 <= results['mAP'] <= 100
    print("PASS: Random retrieval test passed")

def test_ood_metrics_perfect():
    """Test OOD detection with perfect separation"""
    scores_seen = np.ones(100) * 0.9  # High confidence for seen
    scores_unseen = np.ones(100) * 0.1  # Low confidence for unseen
    
    results = compute_ood_metrics(scores_seen, scores_unseen)
    
    assert abs(results['AUROC'] - 1.0) < 0.01, f"AUROC should be 1.0, got {results['AUROC']}"
    assert abs(results['FPR95'] - 0.0) < 0.01, f"FPR95 should be 0.0, got {results['FPR95']}"
    print("PASS: Perfect OOD test passed")

def test_ood_metrics_all_seen():
    """Test OOD detection when all classes seen"""
    scores_seen = np.ones(100) * 0.9
    scores_unseen = np.array([])  # No unseen
    
    results = compute_ood_metrics(scores_seen, scores_unseen)
    
    assert results['AUROC'] is None, f"AUROC should be None, got {results['AUROC']}"
    assert results['FPR95'] is None, f"FPR95 should be None, got {results['FPR95']}"
    print("PASS: All-seen OOD test passed")

def test_lifelong_metrics():
    """Test lifelong learning metrics"""
    # Simulate mAP per task: first task high, then some forgetting
    map_per_task = [80.0, 75.0, 70.0, 65.0]
    num_classes_per_task = [7, 6, 6, 6]
    
    results = compute_lifelong_metrics(map_per_task, num_classes_per_task)
    
    assert results['plasticity'] == 65.0, f"Plasticity should be 65.0, got {results['plasticity']}"
    # Forgetting = avg(80-65, 75-65, 70-65) = avg(15, 10, 5) = 10
    assert abs(results['forgetting'] - 10.0) < 0.01, f"Forgetting should be 10.0, got {results['forgetting']}"
    assert abs(results['overall'] - 55.0) < 0.01, f"Overall should be 55.0, got {results['overall']}"
    print("PASS: Lifelong metrics test passed")

def test_lifelong_single_task():
    """Test lifelong metrics with single task"""
    map_per_task = [80.0]
    num_classes_per_task = [7]
    
    results = compute_lifelong_metrics(map_per_task, num_classes_per_task)
    
    assert results['plasticity'] == 80.0
    assert results['forgetting'] == 0.0
    assert results['overall'] == 80.0
    print("PASS: Single task lifelong test passed")

def test_compute_all_metrics():
    """Test compute_all_metrics integration"""
    num_q, num_g = 20, 50
    q_pids = np.random.randint(0, 10, num_q)
    g_pids = np.random.randint(0, 10, num_g)
    q_camids = np.random.randint(0, 2, num_q)
    g_camids = np.random.randint(0, 2, num_g)
    distmat = np.random.rand(num_q, num_g)
    
    seen = list(range(5))
    unseen = list(range(5, 10))
    map_per_task = [80.0, 75.0]
    num_classes_per_task = [7, 6]
    
    results = compute_all_metrics(
        distmat, q_pids, g_pids, q_camids, g_camids,
        seen_class_ids=seen, unseen_class_ids=unseen,
        map_per_task=map_per_task, num_classes_per_task=num_classes_per_task
    )
    
    assert 'R@1' in results
    assert 'R@5' in results
    assert 'R@10' in results
    assert 'mAP' in results
    assert 'mAP_macro' in results
    assert 'AUROC' in results
    assert 'FPR95' in results
    assert 'Plasticity' in results
    assert 'Forgetting' in results
    assert 'Overall' in results
    print("PASS: compute_all_metrics test passed")

if __name__ == '__main__':
    test_retrieval_metrics_perfect()
    test_retrieval_metrics_random()
    test_ood_metrics_perfect()
    test_ood_metrics_all_seen()
    test_lifelong_metrics()
    test_lifelong_single_task()
    test_compute_all_metrics()
    print("\nAll tests passed!")