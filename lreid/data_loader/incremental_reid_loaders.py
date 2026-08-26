import sys
sys.path.append('../')
import os
from lreid.data_loader.incremental_datasets import IncrementalReIDDataSet, \
    Incremental_combine_train_samples, Incremental_combine_test_samples, IncrementalPersonReIDSamples
import copy
from lreid.datasets import (IncrementalSamples4subcuhksysu, IncrementalSamples4market,
                               IncrementalSamples4duke, IncrementalSamples4sensereid,
                               IncrementalSamples4msmt17, IncrementalSamples4cuhk03,
                               IncrementalSamples4cuhk01, IncrementalSamples4cuhk02,
                               IncrementalSamples4viper, IncrementalSamples4ilids,
                               IncrementalSamples4prid, IncrementalSamples4grid,
                               IncrementalSamples4mix, IncrementalSamples4ip102)
from lreid.data_loader.loader import ClassUniformlySampler4Incremental, data, IterLoader, ClassUniformlySampler
import torch
import torchvision.transforms as transforms
from lreid.data_loader.transforms2 import RandomErasing
from collections import defaultdict


def get_ip102_task_splits(datasets_root):
    """Get IP102 class splits for 4 tasks (7/6/6/6)"""
    import json
    import os.path as osp
    
    # Find dataset root
    candidates = [
        osp.join(datasets_root, 'IP102 dataset/'),
        '/kaggle/input/ip102',
        '/kaggle/input/ip102-dataset',
        '/kaggle/input/IP102',
    ]
    
    json_root = None
    for c in candidates:
        if osp.exists(osp.join(c, 'filtered_class.txt')):
            json_root = c
            break
    
    if json_root is None:
        for root, dirs, files in os.walk(datasets_root):
            if 'filtered_class.txt' in files:
                json_root = root
                break
    
    if json_root is None:
        raise FileNotFoundError('Could not find IP102 filtered_class.txt')
    
    filtered_class_path = osp.join(json_root, 'filtered_class.txt')
    class_ids = []
    with open(filtered_class_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                class_ids.append(int(line))
    
    class_ids = sorted(class_ids)
    assert len(class_ids) == 25, f'Expected 25 classes, got {len(class_ids)}'
    
    # Split 7/6/6/6
    task_splits = [
        class_ids[0:7],    # Task 1: 7 classes
        class_ids[7:13],   # Task 2: 6 classes
        class_ids[13:19],  # Task 3: 6 classes
        class_ids[19:25],  # Task 4: 6 classes
    ]
    
    return task_splits, json_root


class IncrementalReIDLoaders:

    def __init__(self, config):
        self.config = config

        # resize --> flip --> pad+crop --> colorjitor(optional) --> totensor+norm --> rea (optional)
        transform_train = [
            transforms.Resize(self.config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(self.config.image_size)]
        if self.config.use_colorjitor: # use colorjitor
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if self.config.use_rea: # use rea
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        # resize --> totensor --> norm
        self.transform_test = transforms.Compose([
            transforms.Resize(self.config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['market', 'duke', 'cuhksysu', 'subcuhksysu', 'msmt17', 'cuhk03',
                         'mix', 'sensereid',
                         'cuhk01', 'cuhk02', 'viper', 'ilids', 'prid', 'grid', 'generalizable',
                         'allgeneralizable', 'partgeneralizable', 'finalgeneralizable', 'ip102']

        # dataset
        for a_train_dataset in self.config.train_dataset + self.config.test_dataset:
            assert a_train_dataset in self.datasets, a_train_dataset

        # batch size
        self.if_init_show_loader = self.config.output_featuremaps
        self.use_local_label4validation = self.config.use_local_label4validation

        # Check if IP102 is the only dataset for multi-task support
        self.is_ip102_only = (self.config.train_dataset == ['ip102'] and 
                               self.config.test_dataset == ['ip102'])
        
        if self.is_ip102_only:
            # IP102: 4 tasks (7/6/6/6)
            self.ip102_task_splits, self.ip102_json_root = get_ip102_task_splits(self.config.datasets_root)
            self.total_step = 4
        else:
            self.total_step = len(self.config.train_dataset)

        # load
        self._load()
        self._init_device()
        self.continual_train_iter_dict = self.incremental_train_iter_dict


        self.continual_num_pid_per_step = [len(v) for v in self.global_pids_per_step_dict.values()]
        self.continual_num_cid_per_step = [len(v) for v in self.global_cids_per_step_dict.values()]
        print(
            f'Show incremental_num_pid_per_step {self.continual_num_pid_per_step}\n')
        print(
            f'Show incremental_num_cid_per_step {self.continual_num_cid_per_step}\n')
        print(f'Show incremental_train_iter_dict (size = {len(self.continual_train_iter_dict)}): \n {self.continual_train_iter_dict} \n--------end \n')


    def _init_device(self):
        self.device = torch.device('cuda')

    def _load(self):

        '''init train dataset'''
        train_samples = self._get_train_samples(self.config.train_dataset)
        self.incremental_train_iter_dict = {}

        total_pid_list, total_cid_list = [], []
        temp_dict = copy.deepcopy(self.global_pids_per_step_dict)
        for step_index, pid_per_step in self.global_pids_per_step_dict.items():
            if self.config.num_identities_per_domain == -1:
                one_step_pid_list = sorted(list(pid_per_step))
            else:
                one_step_pid_list = sorted(list(pid_per_step))[0:self.config.num_identities_per_domain]
            temp_dict[step_index] = one_step_pid_list
            total_pid_list.extend(one_step_pid_list)
        num_of_real_train = 0
        for item in train_samples:
            if item[1] in total_pid_list:
                num_of_real_train +=1
        print(f'with {self.config.num_identities_per_domain} per domain, the num_of_real_train :{num_of_real_train}')

        for cid_per_step in self.global_cids_per_step_dict.values():
            total_cid_list.extend(cid_per_step)
        del self.global_pids_per_step_dict
        if self.config.joint_train:
            del self.global_cids_per_step_dict
            self.global_pids_per_step_dict = {0: total_pid_list}
            self.global_cids_per_step_dict = {0: total_cid_list}
        else:
            self.global_pids_per_step_dict = temp_dict

        for step_number, one_step_pid_list in self.global_pids_per_step_dict.items():
            self.incremental_train_iter_dict[step_number] = self._get_uniform_incremental_iter(train_samples,
                                                                                                    self.transform_train,
                                                                                                    self.config.p,
                                                                                                    self.config.k,
                                                                                                    one_step_pid_list)


        # self.train_iter = self._get_uniform_iter(train_samples, self.transform_train, self.p, self.k)
        if self.if_init_show_loader:
            self.train_vae_iter = self._get_uniform_iter(train_samples, self.transform_test, 4, 2)
        '''init test dataset'''
        self.test_loader_dict = defaultdict(list)
        query_sample, gallery_sample = [], []
        for one_test_dataset in self.config.test_dataset:
            temp_query_samples, temp_gallery_samples = self._get_test_samples(one_test_dataset)
            query_sample += temp_query_samples
            gallery_sample += temp_gallery_samples
            temp_query_loader = self._get_loader(temp_query_samples, self.transform_test, self.config.test_batch_size)
            temp_gallery_loader = self._get_loader(temp_gallery_samples, self.transform_test,
                                                   self.config.test_batch_size)
            self.test_loader_dict[one_test_dataset].append(temp_query_loader)
            self.test_loader_dict[one_test_dataset].append(temp_gallery_loader)


        IncrementalPersonReIDSamples._show_info(None, train_samples, query_sample, gallery_sample,
                                                name=str(self.config.train_dataset), if_show=True)


    def _get_train_samples(self, train_dataset):
        '''get train samples, support multi-dataset'''
        samples_list = []
        for a_train_dataset in train_dataset:
            if a_train_dataset == 'market':
                samples = IncrementalSamples4market(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'duke':
                samples = IncrementalSamples4duke(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhksysu':
                samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all, use_subset_train=False).train
            elif a_train_dataset == 'subcuhksysu':
                samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all, use_subset_train=True).train
            elif a_train_dataset == 'mix':
                samples = IncrementalSamples4mix(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'sensereid':
                samples = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'msmt17':
                samples = IncrementalSamples4msmt17(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhk03':
                samples = IncrementalSamples4cuhk03(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhk01':
                samples = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'cuhk02':
                samples = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'viper':
                samples = IncrementalSamples4viper(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'ilids':
                samples = IncrementalSamples4ilids(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'prid':
                samples = IncrementalSamples4prid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'grid':
                samples = IncrementalSamples4grid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            elif a_train_dataset == 'ip102':
                samples = IncrementalSamples4ip102(self.config.datasets_root, relabel=True, combineall=self.config.combine_all).train
            samples_list.append(samples)

        # For IP102, we need to create 4 task-specific splits
        if self.is_ip102_only:
            return self._create_ip102_task_samples(samples_list[0])
        else:
            samples, global_pids_per_step_dict, global_cids_per_step_dict = Incremental_combine_train_samples(samples_list)
            self.global_pids_per_step_dict = global_pids_per_step_dict
            self.global_cids_per_step_dict = global_cids_per_step_dict
            return samples

    def _create_ip102_task_samples(self, all_samples):
        """Split IP102 samples into 4 tasks based on class splits"""
        import json
        
        # Get task splits
        task_splits = self.ip102_task_splits  # List of 4 lists, each with original class IDs
        
        # all_samples has format: [img_path, global_pid, camid, dataset_name, original_class_id]
        # We need to group by original_class_id and split into tasks
        
        # Map original class ID to samples
        class_to_samples = defaultdict(list)
        for sample in all_samples:
            orig_class = sample[4]  # original category_id
            class_to_samples[orig_class].append(sample)
        
        # Create task-specific samples with relabeled PIDs (0-based within each task)
        task_samples = []
        global_pids_per_step = {}
        global_cids_per_step = {}
        current_global_pid = 0
        
        for step, orig_classes in enumerate(self.ip102_task_splits):
            task_data = []
            task_pids = set()
            task_cids = set()
            
            # Create local PID mapping for this task
            local_pid_map = {orig_c: i for i, orig_c in enumerate(orig_classes)}
            
            for orig_c in orig_classes:
                for sample in class_to_samples.get(orig_c, []):
                    img_path, _, camid, dataset_name, orig_class = sample
                    local_pid = local_pid_map[orig_c]
                    global_pid = current_global_pid + local_pid
                    # Sample format: [img_path, global_pid, camid, dataset_name, local_pid]
                    # Index 4 = local_pid (0-6 for step 0, 0-5 for step 1, etc.)
                    # This matches the step's classifier output size
                    task_data.append([img_path, global_pid, camid, dataset_name, local_pid])
                    task_pids.add(global_pid)
                    task_cids.add(camid)
            
            global_pids_per_step[step] = task_pids
            global_cids_per_step[step] = task_cids
            current_global_pid += len(orig_classes)
            task_samples.append(task_data)
        
        # Combine all task samples for the combined training data
        combined_samples = []
        for task_data in task_samples:
            combined_samples.extend(task_data)
        
        # Store task-specific info
        self.ip102_task_samples = task_samples
        self.ip102_task_splits = task_splits
        
        self.global_pids_per_step_dict = global_pids_per_step
        self.global_cids_per_step_dict = global_cids_per_step
        
        return combined_samples

    def _get_test_samples(self, a_test_dataset):
        if a_test_dataset == 'market':
            samples = IncrementalSamples4market(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'duke':
            samples = IncrementalSamples4duke(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhksysu':
            samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all,
                                                     use_subset_train=False)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'subcuhksysu':
            samples = IncrementalSamples4subcuhksysu(self.config.datasets_root, relabel=True, combineall=self.config.combine_all,
                                                     use_subset_train=True)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'mix':
            samples = IncrementalSamples4mix(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'sensereid':
            samples = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'msmt17':
            samples = IncrementalSamples4msmt17(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhk03':
            samples = IncrementalSamples4cuhk03(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhk01':
            samples = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'cuhk02':
            samples = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'viper':
            samples = IncrementalSamples4viper(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'ilids':
            samples = IncrementalSamples4ilids(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'prid':
            samples = IncrementalSamples4prid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'grid':
            samples = IncrementalSamples4grid(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
            query, gallery = samples.query, samples.gallery
        elif a_test_dataset == 'generalizable':

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                               combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                               combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(samples_list=[samples4viper,samples4ilids,samples4prid,samples4grid])
        elif a_test_dataset == 'allgeneralizable':

            samples4sensereid = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4cuhk01 = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4cuhk02 = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True,
                                            combineall=self.config.combine_all)

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                            combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(
                samples_list=[samples4viper, samples4ilids, samples4prid, samples4grid,
                              samples4sensereid, samples4cuhk01, samples4cuhk02])
        elif a_test_dataset == 'finalgeneralizable':
            samples4cuhk03 = IncrementalSamples4cuhk03(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4sensereid = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            samples4cuhk01 = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4cuhk02 = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True,
                                            combineall=self.config.combine_all)

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                            combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(
                samples_list=[samples4viper, samples4ilids, samples4prid, samples4grid,
                              samples4sensereid, samples4cuhk01, samples4cuhk02, samples4cuhk03])
        elif a_test_dataset == 'partgeneralizable':

            samples4sensereid = IncrementalSamples4sensereid(self.config.datasets_root, relabel=True,
                                                   combineall=self.config.combine_all)

            # samples4cuhk01 = IncrementalSamples4cuhk01(self.config.datasets_root, relabel=True,
            #                                     combineall=self.config.combine_all)
            #
            # samples4cuhk02 = IncrementalSamples4cuhk02(self.config.datasets_root, relabel=True,
            #                                            combineall=self.config.combine_all)

            samples4viper = IncrementalSamples4viper(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4ilids = IncrementalSamples4ilids(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)

            samples4prid = IncrementalSamples4prid(self.config.datasets_root, relabel=True,
                                            combineall=self.config.combine_all)

            samples4grid = IncrementalSamples4grid(self.config.datasets_root, relabel=True,
                                             combineall=self.config.combine_all)
            query, gallery = Incremental_combine_test_samples(
                samples_list=[samples4viper, samples4ilids, samples4prid, samples4grid,
                              samples4sensereid])
        elif a_test_dataset == 'ip102':
            if self.is_ip102_only:
                # For IP102 multi-task, filter query/gallery per task
                # We'll handle this in the test phase by filtering per task
                samples = IncrementalSamples4ip102(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
                query, gallery = samples.query, samples.gallery
                # Store for per-task filtering
                self.ip102_full_query = query
                self.ip102_full_gallery = gallery
            else:
                samples = IncrementalSamples4ip102(self.config.datasets_root, relabel=True, combineall=self.config.combine_all)
                query, gallery = samples.query, samples.gallery

        return query, gallery


    def get_task_test_samples(self, step):
        """Get filtered query/gallery for a specific IP102 task"""
        if not self.is_ip102_only:
            return None, None
        
        # Get original class IDs up to current step
        seen_orig_classes = set()
        for i in range(step + 1):
            seen_orig_classes.update(self.ip102_task_splits[i])
        
        # Filter query and gallery to only include seen original classes
        # ip102_full_query/gallery have format: [img_path, global_pid, camid, dataset_name, orig_class_id]
        filtered_query = [s for s in self.ip102_full_query if s[4] in seen_orig_classes]
        filtered_gallery = [s for s in self.ip102_full_gallery if s[4] in seen_orig_classes]
        
        return filtered_query, filtered_gallery

    def _get_uniform_incremental_iter(self, samples, transform, p, k, pid_list):
        '''
               load person reid data_loader from images_folder
               and uniformly sample according to class for continual
               '''
        # dataset.sample is list  dataset.transform
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False,
                                 sampler=ClassUniformlySampler4Incremental(dataset, class_position=1, k=k, pid_list=pid_list))
        iters = IterLoader(loader)
        return iters


    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        # dataset.sample is list  dataset.transform
        dataset = IncrementalReIDDataSet(samples,self.total_step, transform=transform)
        # ClassUniformlySampler
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters


    def _get_random_iter(self, samples, transform, batch_size):
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = IncrementalReIDDataSet(samples, self.total_step, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader