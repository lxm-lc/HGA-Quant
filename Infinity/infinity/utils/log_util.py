import os
import os.path as osp
import torch
from datetime import datetime


class QuantStatLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(osp.dirname(log_path), exist_ok=True)
        self.fp = open(log_path, 'a', encoding='utf-8', buffering=1)

    def log(self, message=''):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{timestamp}] {message}'
        print(line, flush=True)
        self.fp.write(line + '\n')
        self.fp.flush()

    def close(self):
        if not self.fp.closed:
            self.fp.close()


def format_bytes(num_bytes):
    if num_bytes is None:
        return 'N/A'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f'{value:.2f} {unit}'
        value /= 1024
    return f'{value:.2f} TB'


def get_path_size(path):
    if not path or not osp.exists(path):
        return None
    if osp.isfile(path):
        return osp.getsize(path)

    total_size = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            file_path = osp.join(root, file_name)
            if osp.islink(file_path):
                continue
            total_size += osp.getsize(file_path)
    return total_size


def get_model_param_count(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())


def get_model_storage_bytes(model):
    if model is None:
        return 0

    total_bytes = 0
    for tensor in model.parameters():
        total_bytes += tensor.numel() * tensor.element_size()
    for tensor in model.buffers():
        if tensor is None:
            continue
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


def get_cuda_memory_stats():
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    return {
        'device_index': device,
        'device_name': torch.cuda.get_device_name(device),
        'allocated': torch.cuda.memory_allocated(device),
        'reserved': torch.cuda.memory_reserved(device),
        'max_allocated': torch.cuda.max_memory_allocated(device),
        'max_reserved': torch.cuda.max_memory_reserved(device),
    }


def log_cuda_memory(logger, prefix):
    stats = get_cuda_memory_stats()
    if stats is None:
        logger.log(f'{prefix} | cuda=unavailable')
        return

    logger.log(
        f'{prefix} | device={stats["device_index"]}:{stats["device_name"]} '
        f'allocated={format_bytes(stats["allocated"])} '
        f'reserved={format_bytes(stats["reserved"])} '
        f'max_allocated={format_bytes(stats["max_allocated"])} '
        f'max_reserved={format_bytes(stats["max_reserved"])}'
    )
