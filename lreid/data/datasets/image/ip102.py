from __future__ import division, print_function, absolute_import
import os
import json
import os.path as osp
from ..dataset import ImageDataset

class IP102(ImageDataset):
    """IP102 Dataset for pest classification.

    Dataset statistics:
        - identities: 25 classes (from filtered_class.txt)
        - images: ~9500 train + ~2000 val + ~7000 test
    """
    dataset_dir = 'IP102 dataset'
    
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self._find_dataset_root(self.root)
        
        self.train_json = osp.join(self.dataset_dir, 'train.json')
        self.val_json = osp.join(self.dataset_dir, 'val.json')
        self.test_json = osp.join(self.dataset_dir, 'test.json')
        self.classes_txt = osp.join(self.dataset_dir, 'classes.txt')
        self.filtered_class_txt = osp.join(self.dataset_dir, 'filtered_class.txt')
        
        self.class_id_to_name = self._load_class_names()
        self.valid_class_ids = self._load_filtered_classes()
        self.images_dir = self._find_images_dir(self.dataset_dir)
        
        train = self.process_json(self.train_json, relabel=True)
        query = self.process_json(self.val_json, relabel=False)
        gallery = self.process_json(self.test_json, relabel=False)
        
        super(IP102, self).__init__(train, query, gallery, **kwargs)
    
    def _find_dataset_root(self, root):
        candidates = [
            osp.join(root, self.dataset_dir),
            osp.join(root, 'IP102_dataset'),
            osp.join(root, 'ip102'),
            '/kaggle/input/ip102',
            '/kaggle/input/ip102-dataset',
            '/kaggle/input/IP102',
        ]
        
        for c in candidates:
            if osp.exists(osp.join(c, 'train.json')):
                return c
        
        for r, dirs, files in os.walk(root):
            if 'train.json' in files and 'classes.txt' in files:
                return r
        
        raise FileNotFoundError(f'Could not find IP102 dataset. Searched in: {candidates}')
    
    def _find_images_dir(self, json_root):
        """Find the directory containing the actual image files"""
        candidates = [
            osp.join(json_root, 'VOC2007', 'VOC2007', 'JPEGImages'),
            osp.join(json_root, 'JPEGImages'),
            osp.join(json_root, 'images'),
            osp.join(osp.dirname(json_root), 'VOC2007', 'VOC2007', 'JPEGImages'),
            osp.join(osp.dirname(json_root), 'JPEGImages'),
            osp.join(osp.dirname(json_root), 'images'),
            json_root,
        ]
        
        for c in candidates:
            if osp.exists(c) and any(f.endswith('.jpg') for f in os.listdir(c)[:5]):
                return c
        
        # Fallback: search recursively
        for root, dirs, files in os.walk(json_root):
            jpg_files = [f for f in files if f.endswith('.jpg')]
            if len(jpg_files) > 100:
                return root
        
        return json_dir
    
    def _load_class_names(self):
        mapping = {}
        if osp.exists(self.classes_txt):
            with open(self.classes_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            class_id, name = parts
                            mapping[int(class_id)] = name.strip()
        return mapping
    
    def _load_filtered_classes(self):
        class_ids = []
        if osp.exists(self.filtered_class_txt):
            with open(self.filtered_class_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_ids.append(int(line))
        return sorted(class_ids)
    
    def process_json(self, json_path, relabel=False):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        
        valid_class_ids = set(self.valid_class_ids)
        class_id_to_label = {cid: i for i, cid in enumerate(self.valid_class_ids)}
        
        data_list = []
        for ann in annotations:
            category_id = ann['category_id']
            if category_id not in valid_class_ids:
                continue
            
            image_info = images.get(ann['image_id'])
            if not image_info:
                continue
            
            file_name = image_info['file_name']
            img_path = osp.join(self.images_dir, file_name)
            
            if not osp.exists(img_path):
                # Fallback search
                alt_paths = [
                    osp.join(osp.dirname(json_path), file_name),
                    osp.join(osp.dirname(json_path), 'images', file_name),
                    osp.join(osp.dirname(osp.dirname(json_path)), 'images', file_name),
                    osp.join(osp.dirname(osp.dirname(json_path)), 'VOC2007', 'VOC2007', 'JPEGImages', file_name),
                    osp.join(osp.dirname(osp.dirname(json_path)), 'JPEGImages', file_name),
                ]
                for alt in alt_paths:
                    if osp.exists(alt):
                        img_path = alt
                        break
            
            if relabel:
                pid = class_id_to_label[category_id]
            else:
                pid = category_id
            
            camid = 0
            data_list.append((img_path, pid, camid))
        
        return data_list