import torch
import torch.nn as nn
import os

def unwrap_model(model):
    """Unwrap model from DataParallel/DistributedDataParallel if needed"""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model

def get_loader_kwargs(batch_size, num_gpus, drop_last=True, num_workers=8):
    """Get DataLoader kwargs for multi-GPU training
    
    Ensures batch_size is divisible by num_gpus and sets drop_last=True
    """
    if num_gpus > 1:
        assert batch_size % num_gpus == 0, \
            f"batch_size ({batch_size}) must be divisible by num_gpus ({num_gpus})"
        effective_batch_size = batch_size // num_gpus
    else:
        effective_batch_size = batch_size
    
    return {
        'batch_size': effective_batch_size,
        'num_workers': num_workers,
        'drop_last': drop_last,
        'pin_memory': True
    }

def get_num_gpus(device_config=None):
    """Get number of GPUs from config or auto-detect"""
    if device_config is not None:
        if isinstance(device_config, (list, tuple)):
            return len(device_config)
        elif isinstance(device_config, str):
            return len(device_config.split(','))
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if devices:
            return len(devices.split(','))
    
    return torch.cuda.device_count()

def setup_multi_gpu(model, device_config=None):
    """Setup model for multi-GPU training"""
    num_gpus = get_num_gpus(device_config)
    
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        if device_config is not None:
            if isinstance(device_config, str):
                device_ids = [int(d) for d in device_config.split(',')]
            elif isinstance(device_config, (list, tuple)):
                device_ids = [int(d) for d in device_config]
        
        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
    else:
        model = model.cuda()
    
    return model, num_gpus

def get_device(device_config=None):
    """Get torch device"""
    num_gpus = get_num_gpus(device_config)
    if num_gpus > 0:
        return torch.device('cuda')
    return torch.device('cpu')