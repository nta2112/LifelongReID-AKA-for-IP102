import os
import glob

def find_dataset_root(dataset_name, base_paths=None, env_var=None):
    """Auto-discover dataset root path
    
    Args:
        dataset_name: Name of dataset (e.g., 'IP102')
        base_paths: List of base paths to search
        env_var: Environment variable name that may contain the path
    
    Returns:
        Path to dataset root directory
    """
    candidates = []
    
    if env_var and env_var in os.environ:
        candidates.append(os.environ[env_var])
    
    if base_paths:
        candidates.extend(base_paths)
    
    candidates.extend([
        '/kaggle/input',
        '/kaggle/input/ip102',
        '/kaggle/input/ip102-dataset',
        '/kaggle/input/IP102',
        os.path.expanduser('~/datasets'),
        os.path.expanduser('~/data'),
        '/data',
        '/datasets',
    ])
    
    for base in candidates:
        if not base or not os.path.exists(base):
            continue
        
        for root, dirs, files in os.walk(base):
            if 'train.json' in files and 'classes.txt' in files:
                if dataset_name.lower() == 'ip102':
                    if 'filtered_class.txt' in files:
                        return root
                else:
                    return root
    
    raise FileNotFoundError(
        f'Could not find {dataset_name} dataset. '
        f'Searched in: {candidates}'
    )

def find_file(filename, search_paths=None):
    """Find a file by searching through paths"""
    if search_paths is None:
        search_paths = [
            os.getcwd(),
            os.path.expanduser('~'),
            '/kaggle/input',
            '/data',
            '/datasets',
        ]
    
    for base in search_paths:
        if not os.path.exists(base):
            continue
        
        for root, dirs, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    
    return None

def get_ip102_class_files(datasets_root=None):
    """Get paths to IP102 class files (classes.txt and filtered_class.txt)"""
    if datasets_root is None:
        datasets_root = find_dataset_root('IP102', env_var='IP102_DATA_ROOT')
    
    classes_txt = os.path.join(datasets_root, 'classes.txt')
    filtered_class_txt = os.path.join(datasets_root, 'filtered_class.txt')
    
    if not os.path.exists(classes_txt):
        classes_txt = find_file('classes.txt')
    if not os.path.exists(filtered_class_txt):
        filtered_class_txt = find_file('filtered_class.txt')
    
    return classes_txt, filtered_class_txt

def load_ip102_class_mapping(classes_txt=None, filtered_class_txt=None):
    """Load IP102 class mappings"""
    if classes_txt is None or filtered_class_txt is None:
        classes_txt, filtered_class_txt = get_ip102_class_files()
    
    class_id_to_name = {}
    if os.path.exists(classes_txt):
        with open(classes_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        class_id, name = parts
                        class_id_to_name[int(class_id)] = name.strip()
    
    valid_class_ids = []
    if os.path.exists(filtered_class_txt):
        with open(filtered_class_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    valid_class_ids.append(int(line))
    
    valid_class_ids = sorted(valid_class_ids)
    
    return class_id_to_name, valid_class_ids

def get_task_split(valid_class_ids, split_ratios=None):
    """Split 25 classes into 4 tasks (7/6/6/6 by default)"""
    if split_ratios is None:
        split_ratios = [7, 6, 6, 6]
    
    assert sum(split_ratios) == len(valid_class_ids), \
        f"Split ratios {split_ratios} don't match {len(valid_class_ids)} classes"
    
    tasks = []
    start = 0
    for ratio in split_ratios:
        tasks.append(valid_class_ids[start:start + ratio])
        start += ratio
    
    return tasks