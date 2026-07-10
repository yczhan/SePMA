import os
import random
import numpy as np
from PIL import Image
import io
import zipfile
from decord import VideoReader, cpu, gpu
import torch
import torch.utils.data
import torch.utils.dlpack as dlpack
import utils.logging as logging
from collections import OrderedDict

import time
import traceback

import torchvision
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo,
    KineticsResizedCrop,
    KineticsResizedCropFewshot
)
# from datasets.utils.shuffle import shuffle
# from datasets.utils.unfold import unfold

from datasets.base.base_dataset import BaseVideoDataset

import utils.bucket as bu
import glob
from datasets.base.builder import DATASET_REGISTRY
from datasets.utils.random_erasing import RandomErasing

logger = logging.get_logger(__name__)


class Split_few_shot():
    """Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
    def __init__(self, folder, split_dataset='train', dataset="Ssv2_few_shot"):
        # self.args = args
        
        self.gt_a_list = []
        self.videos = []
        self.split_dataset = split_dataset

        if dataset == 'Ssv2_few_shot':

            for class_folder in folder:
                paths = class_folder.strip().split('/')[-1]

                class_id = int(class_folder.strip().split('/')[0][len(split_dataset):]) # class_folders.index(class_folder)
                self.add_vid(paths, class_id)
        else:
            for class_folder in folder:
                paths = class_folder.strip().split('//')[-1]

                class_id = int(class_folder.strip().split('//')[0][len(split_dataset):]) # class_folders.index(class_folder)
                self.add_vid(paths, class_id)

        logger.info("loaded {} videos from {} dataset: {} !".format(len(self.gt_a_list), split_dataset, dataset))

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_single_video(self, index):
        return self.videos[index], self.gt_a_list[index]

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def __len__(self):
        return len(self.gt_a_list)


@DATASET_REGISTRY.register()
class Ssv2_few_shot(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Ssv2_few_shot, self).__init__(cfg, split) 
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        # if hasattr(self.cfg.TRAIN, "DATASET_FEW"):
        #     self.dataset_name = self.cfg.TRAIN.DATASET_FEW
        self.split_dataset = split

    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """

        name = "{}_few_shot.txt".format(   
            "train" if self.split == "train" else "test",
        )
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        class_ = self._samples[index]["label_idx"]
        video_path = os.path.join(self.data_root_dir, self._samples[index]["id"]+".mp4")
        sample_info = {
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info
    

    def _construct_dataset(self, cfg):

        if hasattr(self.cfg.TRAIN, "DATASET_FEW"):
            self.dataset_name = self.cfg.TRAIN.DATASET_FEW
        
        self._num_clips = 1

        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        for retry in range(5):
            try:
                logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
                local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
                local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
                if local_file[-4:] == ".csv":
                    import pandas
                    lines = pandas.read_csv(local_file)
                    for line in lines.values.tolist():
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                elif local_file[-4:] == "json":
                    import json
                    with open(local_file, "r") as f:
                        lines = json.load(f)
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                else:
                    with open(local_file) as f:
                        lines = f.readlines()
                        for line in lines:
                            for idx in range(self._num_clips):
                                self._samples.append(line.strip())
                                self._spatial_temporal_index.append(idx)
                self.split_few_shot = Split_few_shot(lines, self.split, dataset=self.dataset_name)
                logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
                break
            except:
                if retry<4:
                    continue
                else:
                    raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))

        if hasattr(self.cfg.TRAIN, "FEW_SHOT") and self.cfg.TRAIN.FEW_SHOT and self.split == "train":
            """ Sample number setting for training in few-shot settings: 
                During few shot training, the batch size could be larger than the size of the training samples.
                Therefore, the number of samples in the same sample is multiplied by 10 times, and the training schedule is reduced by 10 times. 
            """
            self._samples = self._samples * 10
            print("10 FOLD FEW SHOT SAMPLES")
            
        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))


    def __getitem__(self, index):
        """
            Returns:
                frames (dict): {
                    "video": (tensor), 
                    "text_embedding" (optional): (tensor)
                }
                labels (dict): {
                    "supervised": (tensor),
                    "self-supervised" (optional): (...)
                }
        """
        if self.cfg.TRAIN.META_BATCH:
            """returns dict of support and target images and labels for a meta training task"""
            #select classes to use for this task
            c = self.split_few_shot
            classes = c.get_unique_classes()
            batch_classes = random.sample(classes, self.cfg.TRAIN.WAY)
            if self.split != "train" and hasattr(self.cfg.TRAIN, "WAT_TEST"):
                batch_classes = random.sample(classes, self.cfg.TRAIN.WAT_TEST)

            if self.split == "train":
                n_queries = self.cfg.TRAIN.QUERY_PER_CLASS
                SHOT = self.cfg.TRAIN.SHOT
            else:
                n_queries = self.cfg.TRAIN.QUERY_PER_CLASS_TEST
                if hasattr(self.cfg.TRAIN, "SHOT_TEST"):
                    SHOT = self.cfg.TRAIN.SHOT_TEST
                else:
                    SHOT = self.cfg.TRAIN.SHOT
            
            retries = 5
            for retry in range(retries):
                try:
                    support_set = []
                    support_labels = []
                    target_set = []
                    target_labels = []
                    real_support_labels = []
                    real_target_labels = []

                    for bl, bc in enumerate(batch_classes):
                        n_total = c.get_num_videos_for_class(bc)
                        # retries = 5
                        # for retry in range(retries):
                        #     try:
                        idxs = random.sample([i for i in range(n_total)], SHOT + n_queries)

                        for idx in idxs[0:SHOT]:
                            if hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF_SUPPORT") and self.cfg.AUGMENTATION.SUPPORT_QUERY_DIFF_SUPPORT and self.split_dataset=="train":
                                vid, vid_id = self.get_seq_query(bc, idx)
                            else:
                                vid, vid_id = self.get_seq(bc, idx)
                            support_set.append(vid)
                            support_labels.append(bl)
                            real_support_labels.append(bc)
                        
                        # try:
                        for idx in idxs[SHOT:]:
                            if hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF") and self.cfg.AUGMENTATION.SUPPORT_QUERY_DIFF and self.split_dataset=="train":
                                vid, vid_id = self.get_seq_query(bc, idx)
                            else:
                                vid, vid_id = self.get_seq(bc, idx)
                            target_set.append(vid)
                            target_labels.append(bl)
                            real_target_labels.append(bc)
                    break
                            
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, idxs, bc
                    ))
            
            s = list(zip(support_set, support_labels, real_support_labels))
            random.shuffle(s)
            support_set, support_labels, real_support_labels = zip(*s)
            
            t = list(zip(target_set, target_labels, real_target_labels))
            random.shuffle(t)
            target_set, target_labels, real_target_labels = zip(*t)
            
            support_set = torch.cat(support_set)  # [200, 3, 224, 224]

            target_set = torch.cat(target_set)    # [200, 3, 224, 224]
            support_labels = torch.FloatTensor(support_labels)
            target_labels = torch.FloatTensor(target_labels)
            real_target_labels = torch.FloatTensor(real_target_labels)  # shape: [25]
            real_support_labels = torch.FloatTensor(real_support_labels)
            # [45., 59., 45., 11., 39., 39., 39., 11., 11., 25., 25., 25., 59., 45., 11., 25., 59., 25., 45., 39., 45., 59., 39., 59., 11.]
            batch_classes = torch.FloatTensor(batch_classes) # [45., 11., 59., 25., 39.]
            
            return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes, "real_support_labels":real_support_labels}

        else:
            sample_info = self._get_sample_info(index)

            retries = 1 if self.split == "train" else 10
            for retry in range(retries):
                try:
                    data, file_to_remove, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, index, sample_info["path"]
                    ))

            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
                if not hasattr(self, "label_embd"):
                    self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
                data["text_embedding"] = self.label_embd

            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()

            # self.visualize_frames(data["video"], index)
            
            labels = {}
            labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
            if self.cfg.PRETRAIN.ENABLE:
                try:
                    data, labels["self-supervised"] = self.ssl_generator(data, index)
                except Exception as e:
                    traceback.print_exc()
                    print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                        index, sample_info["path"], data["video"].shape
                    ))
                    return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            else:
                labels["self-supervised"] = {}
                if "flow" in data.keys() and "video" in data.keys():
                    data = self.transform(data)
                elif "video" in data.keys():
                    data["video"] = self.transform(data["video"]) # C, T, H, W = 3, 16, 240, 320, RGB
                    
            if  "Slowfast" in self.cfg.VIDEO.BACKBONE.META_ARCH and self.split not in ['extract_feat']:
                slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1]//self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA+1).long()[:-1]
                fast_frames = data["video"].clone()
                slow_frames = data["video"][:,slow_idx,:,:].clone()
                data["video"] = [slow_frames, fast_frames]
            bu.clear_tmp_file(file_to_remove)

            if self.split in ['extract_feat']:
                meta = {'video_name': sample_info['video_name'],
                        'subset': sample_info['subset']}
            else:
                meta = {}

            # self.reversed_visualize_frames(data["video"], index)
            return data, labels, index, meta

    def get_seq(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.split_few_shot
        if self.cfg.TRAIN.META_BATCH:
            paths, vid_id = c.get_rand_vid(label, idx) 
            # imgs = self.load_and_transform_paths(paths)
            if self.dataset_name == 'Ssv2_few_shot':
                video_path = os.path.join(self.data_root_dir, paths + ".webm")
            else:
                video_path = os.path.join(self.data_root_dir, paths)
                if self.dataset_name == 'Kinetics_few_shot':
                    dirs = video_path.split("/")
                    folder_name = dirs[-2]
                    new_folder_name = folder_name.replace("_", " ")
                    video_path = video_path.replace(folder_name, new_folder_name)
            sample_info = {
                "path": video_path,
                # "supervised_label": class_,
            }
            # sample_info = self._get_sample_info(index)
            index = vid_id
            retries = 5 if self.split == "train" else 10   # 1
            for retry in range(retries):
                try:
                    data, file_to_remove, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, index, sample_info["path"]
                    ))

            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            # if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
            #     if not hasattr(self, "label_embd"):
            #         self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
            #     data["text_embedding"] = self.label_embd

            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()

            # self.visualize_frames(data["video"], index)
            
            labels = {}
            labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
            if self.cfg.PRETRAIN.ENABLE:
                try:
                    data, labels["self-supervised"] = self.ssl_generator(data, index)
                except Exception as e:
                    traceback.print_exc()
                    print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                        index, sample_info["path"], data["video"].shape
                    ))
                    return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            else:
                labels["self-supervised"] = {}
                if "flow" in data.keys() and "video" in data.keys():
                    data = self.transform(data)
                elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                    data["video"] = self.transform(data["video"]) # C, T, H, W = 3, 16, 240, 320, RGB
                    
            if  "Slowfast" in self.cfg.VIDEO.BACKBONE.META_ARCH and self.split not in ['extract_feat']:
                slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1]//self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA+1).long()[:-1]
                fast_frames = data["video"].clone()
                slow_frames = data["video"][:,slow_idx,:,:].clone()
                data["video"] = [slow_frames, fast_frames]
            bu.clear_tmp_file(file_to_remove)
            
            return data["video"].permute(1,0,2,3), vid_id
    
    def get_seq_query(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.split_few_shot
        if self.cfg.TRAIN.META_BATCH:
            paths, vid_id = c.get_rand_vid(label, idx) 
            # imgs = self.load_and_transform_paths(paths)
            if self.dataset_name == 'Ssv2_few_shot':
                video_path = os.path.join(self.data_root_dir, paths + ".webm")
            else:
                video_path = os.path.join(self.data_root_dir, paths)
                if self.dataset_name == 'Kinetics_few_shot':
                    dirs = video_path.split("/")
                    folder_name = dirs[-2]
                    new_folder_name = folder_name.replace("_", " ")
                    video_path = video_path.replace(folder_name, new_folder_name)
            sample_info = {
                "path": video_path,
                # "supervised_label": class_,
            }
            # sample_info = self._get_sample_info(index)
            index = vid_id
            retries = 5 if self.split == "train" else 10   # 1
            for retry in range(retries):
                try:
                    data, file_to_remove, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, index, sample_info["path"]
                    ))

            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            # if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
            #     if not hasattr(self, "label_embd"):
            #         self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
            #     data["text_embedding"] = self.label_embd

            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()

            labels = {}
            labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
            if self.cfg.PRETRAIN.ENABLE:
                try:
                    data, labels["self-supervised"] = self.ssl_generator(data, index)
                except Exception as e:
                    traceback.print_exc()
                    print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                        index, sample_info["path"], data["video"].shape
                    ))
                    return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            else:
                labels["self-supervised"] = {}
                if "flow" in data.keys() and "video" in data.keys():
                    data = self.transform(data)
                elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                    data["video"] = self.transform_query(data["video"]) # C, T, H, W = 3, 16, 240, 320, RGB

            bu.clear_tmp_file(file_to_remove)
            
            return data["video"].permute(1,0,2,3), vid_id


    def __len__(self):
        if hasattr(self.cfg.TRAIN, "META_BATCH") and self.split == 'train' and self.cfg.TRAIN.META_BATCH:
            return self.cfg.TRAIN.NUM_SAMPLES
        elif hasattr(self.cfg.TRAIN, "NUM_TEST_TASKS") and self.cfg.TRAIN.NUM_TEST_TASKS:
            return self.cfg.TRAIN.NUM_TEST_TASKS
        else:
            return len(self.split_few_shot)  # len(self._samples)
    

    def _config_transform(self):
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list_query = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]  
            if hasattr(self.cfg.AUGMENTATION, "RANDOM_FLIP") and self.cfg.AUGMENTATION.RANDOM_FLIP:
                std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]     # KineticsResizedCrop
            else:
                std_transform_list = [
                    transforms.ToTensorVideo(),
                    KineticsResizedCropFewshot(
                        short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                        crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                    ),
                    # transforms.RandomHorizontalFlipVideo()
                ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
                ]

            if hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_ERASE") and self.cfg.AUGMENTATION.NO_RANDOM_ERASE:
                std_transform_list += [
                    transforms.NormalizeVideo(
                        mean=self.cfg.DATA.MEAN,
                        std=self.cfg.DATA.STD,
                        inplace=True
                    ),
                    # RandomErasing(self.cfg)
                ]
            else:
                std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
                ]
            self.transform = Compose(std_transform_list)
            self.transform_query = Compose(std_transform_list_query)
        elif self.split == 'val' or self.split == 'test':
            idx = -1
            if hasattr(self.cfg.DATA, "TEST_CENTER_CROP"):
                idx = self.cfg.DATA.TEST_CENTER_CROP
            
            if isinstance(self.cfg.DATA.TEST_SCALE, list):
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TEST_SCALE[0], self.cfg.DATA.TEST_SCALE[1]],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS,
                    idx = idx
                )   # KineticsResizedCrop
            else:
                self.resize_video = KineticsResizedCropFewshot(
                        short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                        crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                        num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS,
                        idx = idx
                    )   # KineticsResizedCrop
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)


    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, clip_idx, num_clips, num_frames, interval)


class Split():
    """Contains video frame paths and ground truth labels for a single split (e.g. train videos).
    """

    def __init__(self, args):
        self.args = args

        self.gt_a_list = []  # [0, 0, 0, 1, 1, 4, ...]  global class label, may not be continuous
        self.videos = []  # [ [img0, img1, img2,...], [img0, img1, ...], ... ]

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)

        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_single_video(self, index):
        return self.videos[index], self.gt_a_list[index]

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


@DATASET_REGISTRY.register()
class FSARVideoDataset(torch.utils.data.Dataset):
    """Dataset for few-shot videos, which returns few-shot tasks. """
    def __init__(self, args, meta_batches=True):
        self.args = args
        self.get_item_counter = 0
        self.meta_batches = True

        self.data_dir = args.DATA.DATA_ROOT_DIR

        self.seq_len = args.DATA.NUM_INPUT_FRAMES
        self.split = "train"
        self.tensor_transform = torchvision.transforms.ToTensor()
        self.img_size = args.DATA.TRAIN_CROP_SIZE

        self.annotation_path = args.annotation_path if hasattr(args, "annotation_path") else args.DATA.ANNO_DIR

        self.way = args.TRAIN.WAY
        self.shot = args.TRAIN.SHOT
        self.query_per_class = args.TRAIN.QUERY_PER_CLASS

        self.train_split = Split(self.args)
        self.val_split = Split(self.args)
        self.test_split = Split(self.args)

        # self.hyrsm_transforms()
        self._config_transform()
        self._select_fold()
        start = time.time()
        self.read_lists()

        end = time.time()
        print('read the dir cost time:', end - start)

    def hyrsm_transforms(self):
        print('Setup transforms according to HyRSM')
        std_transform_list_query = [
            # transforms.ToTensorVideo(),   # int (T, H, W, C) -> float (C, T, H, W)
            transforms.RandomHorizontalFlipVideo(),
            KineticsResizedCropFewshot(short_side_range=[256, 256], crop_size=self.args.DATA.TRAIN_CROP_SIZE, ), ]
        if hasattr(self.args.AUGMENTATION, "RANDOM_FLIP") and self.args.AUGMENTATION.RANDOM_FLIP:
            std_transform_list = [
                # transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range=[256, 256],
                    crop_size=self.args.DATA.TRAIN_CROP_SIZE,
                ), ]
        else:
            std_transform_list = [
                # transforms.ToTensorVideo(),
                KineticsResizedCropFewshot(
                    short_side_range=[256, 256],
                    crop_size=self.args.DATA.TRAIN_CROP_SIZE,
                ), ]
        # Add color aug
        if self.args.AUGMENTATION.COLOR_AUG:
            std_transform_list.append(
                ColorJitter(
                    brightness=self.args.AUGMENTATION.BRIGHTNESS,
                    contrast=self.args.AUGMENTATION.CONTRAST,
                    saturation=self.args.AUGMENTATION.SATURATION,
                    hue=self.args.AUGMENTATION.HUE,
                    grayscale=self.args.AUGMENTATION.GRAYSCALE,
                    consistent=self.args.AUGMENTATION.CONSISTENT,
                    shuffle=self.args.AUGMENTATION.SHUFFLE,
                    gray_first=self.args.AUGMENTATION.GRAY_FIRST,
                    is_split=self.args.AUGMENTATION.IS_SPLIT
                ),
            )
        std_transform_list_query.append(
            ColorJitter(
                brightness=self.args.AUGMENTATION.BRIGHTNESS,
                contrast=self.args.AUGMENTATION.CONTRAST,
                saturation=self.args.AUGMENTATION.SATURATION,
                hue=self.args.AUGMENTATION.HUE,
                grayscale=self.args.AUGMENTATION.GRAYSCALE,
                consistent=self.args.AUGMENTATION.CONSISTENT,
                shuffle=self.args.AUGMENTATION.SHUFFLE,
                gray_first=self.args.AUGMENTATION.GRAY_FIRST,
                is_split=self.args.AUGMENTATION.IS_SPLIT
            ),
        )
        std_transform_list_query += [
            transforms.NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], inplace=True),
            RandomErasing(self.args)
        ]
        std_transform_list += [
            transforms.NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], inplace=True),
        ]
        if not (hasattr(self.args.AUGMENTATION, "NO_RANDOM_ERASE") and self.args.AUGMENTATION.NO_RANDOM_ERASE):
            std_transform_list += [RandomErasing(self.args)]

        self.transform = {}
        self.transform["train"] = Compose(std_transform_list)
        self.transform_query = Compose(std_transform_list_query)

        std_transform_test_list = [
            # transforms.ToTensorVideo(),
            KineticsResizedCropFewshot(short_side_range=[256, 256], crop_size=self.args.DATA.TRAIN_CROP_SIZE, idx=True),
            transforms.NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], inplace=True)
        ]
        self.transform["test"] = Compose(std_transform_test_list)

    def _config_transform(self):
        self.cfg = self.args
        self.transform = None

        std_transform_list_query = [
            transforms.ToTensorVideo(),
            transforms.RandomHorizontalFlipVideo(),
            KineticsResizedCropFewshot(
                short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
            ), ]
        if hasattr(self.cfg.AUGMENTATION, "RANDOM_FLIP") and self.cfg.AUGMENTATION.RANDOM_FLIP:
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
                ), ]  # KineticsResizedCrop
        else:
            std_transform_list = [
                transforms.ToTensorVideo(),
                KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
                ),
                # transforms.RandomHorizontalFlipVideo()
            ]
        # Add color aug
        if self.cfg.AUGMENTATION.COLOR_AUG:
            std_transform_list.append(
                ColorJitter(
                    brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                    contrast=self.cfg.AUGMENTATION.CONTRAST,
                    saturation=self.cfg.AUGMENTATION.SATURATION,
                    hue=self.cfg.AUGMENTATION.HUE,
                    grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                    consistent=self.cfg.AUGMENTATION.CONSISTENT,
                    shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                    gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                    is_split=self.cfg.AUGMENTATION.IS_SPLIT
                ),
            )
        std_transform_list_query.append(
            ColorJitter(
                brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                contrast=self.cfg.AUGMENTATION.CONTRAST,
                saturation=self.cfg.AUGMENTATION.SATURATION,
                hue=self.cfg.AUGMENTATION.HUE,
                grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                consistent=self.cfg.AUGMENTATION.CONSISTENT,
                shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                is_split=self.cfg.AUGMENTATION.IS_SPLIT
            ),
        )
        std_transform_list_query += [
            transforms.NormalizeVideo(
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
                inplace=True
            ),
            RandomErasing(self.cfg)
        ]

        if hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_ERASE") and self.cfg.AUGMENTATION.NO_RANDOM_ERASE:
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                # RandomErasing(self.cfg)
            ]
        else:
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
            ]

        self.transform = {}
        self.transform["train"] = Compose(std_transform_list)
        self.transform_query = Compose(std_transform_list_query)

        idx = -1
        if hasattr(self.cfg.DATA, "TEST_CENTER_CROP"):
            idx = self.cfg.DATA.TEST_CENTER_CROP

        if isinstance(self.cfg.DATA.TEST_SCALE, list):
            self.resize_video = KineticsResizedCropFewshot(
                short_side_range=[self.cfg.DATA.TEST_SCALE[0], self.cfg.DATA.TEST_SCALE[1]],
                crop_size=self.cfg.DATA.TEST_CROP_SIZE,
                num_spatial_crops=self.cfg.TEST.NUM_SPATIAL_CROPS,
                idx=idx
            )  # KineticsResizedCrop
        else:
            self.resize_video = KineticsResizedCropFewshot(
                short_side_range=[self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                crop_size=self.cfg.DATA.TEST_CROP_SIZE,
                num_spatial_crops=self.cfg.TEST.NUM_SPATIAL_CROPS,
                idx=idx
            )  # KineticsResizedCrop
        std_transform_list = [
            transforms.ToTensorVideo(),
            self.resize_video,
            transforms.NormalizeVideo(
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
                inplace=True
            )
        ]
        self.transform["test"] = Compose(std_transform_list)

    '''read the train/val/test lists and load the paths/labels from the dataset dir. In case data_dir is much larger
    than the videos in the lists so that read_dir() will waste lots of time.'''
    def read_lists(self):
        print('self.data_dir', self.data_dir, ' , read from the lists')
        if 'kinetics' in self.data_dir.lower():
            self.data_dir = os.path.join(self.data_dir, 'train/')

        class_folders = os.listdir(self.data_dir)
        class_folders.sort()
        class_folders_exist = True
        if len(class_folders) > 1000:
            print('More than 1000 folders in the current dir. Means no class folders above video folders')
            class_folders_exist = False
        split_dict = {'train': self.train_split, 'val': self.val_split, 'test': self.test_split}
        for name in ["train", "val", "test"]:
            c = split_dict[name]
            for video_folder in self.train_test_lists[name]:
                class_name = self.train_test_dict_clsname[name][video_folder]
                if class_folders_exist:
                    video_frame_dir = os.path.join(self.data_dir, class_name, video_folder)
                else:
                    video_frame_dir = os.path.join(self.data_dir, video_folder)
                if os.path.exists(video_frame_dir):
                    if 'i' in os.listdir(video_frame_dir):  # sometimes there are i, x, y subfolders
                        video_frame_dir = os.path.join(video_frame_dir, 'i')
                    imgs = os.listdir(video_frame_dir)
                    if len(imgs) < self.seq_len:
                        continue
                    imgs.sort()
                    paths = [os.path.join(video_frame_dir, img) for img in imgs]
                    paths.sort()
                    class_id = self.class_name_unique.index(class_name)
                    c.add_vid(paths, class_id)
                elif os.path.exists(video_frame_dir + '.avi'):
                    video_frame_dir = video_frame_dir + '.avi'
                    class_id = self.class_name_unique.index(class_name)
                    c.add_vid(video_frame_dir, class_id)
                elif os.path.exists(video_frame_dir+'.webm'):
                    video_frame_dir = video_frame_dir + '.webm'
                    class_id = self.class_name_unique.index(class_name)
                    c.add_vid(video_frame_dir, class_id)
                elif os.path.exists(video_frame_dir + '.mp4'):
                    video_frame_dir = video_frame_dir + '.mp4'
                    class_id = self.class_name_unique.index(class_name)
                    c.add_vid(video_frame_dir, class_id)
                elif os.path.exists(video_frame_dir.replace('/train/', '/val/') + '.mp4'):
                    video_frame_dir = video_frame_dir.replace('/train/', '/val/') + '.mp4'
                    class_id = self.class_name_unique.index(class_name)
                    c.add_vid(video_frame_dir, class_id)
                else:
                    print('Warning! video {} is not found in data_dir'.format(video_frame_dir))

        print("loaded {}".format(self.data_dir))
        print("train: {}, val: {}, test: {}".format(len(self.train_split), len(self.val_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_or_test_db(self, split=None):
        if split is None:
            return self.get_split()
        else:
            if split in self.train_test_lists["train"]:
                return self.train_split
            elif split in self.train_test_lists["test"]:
                return self.test_split
            elif split in self.train_test_lists["val"]:
                return self.val_split
            else:
                # print('Warning! This video is not found in train/val/test lists')
                return None

    """ load the paths of all videos in the train/val/test lists. """
    def _select_fold(self):
        lists = {}
        dict_clsname = {}  # store mappings dict of video to label.
        class_name_unique = []
        class_name_prefix = 'full_' if 'ssv2_full' in self.annotation_path else 'small_' if 'ssv2_small' in self.annotation_path else ''
        for name in ["train", "test", "val"]:
            fname = "{}.txt".format(name)
            f = os.path.join(self.annotation_path, fname)

            pattern = "{}*.txt".format(name)
            search_path = os.path.join(self.annotation_path, pattern)
            matching_files = glob.glob(search_path)

            if matching_files:
                f = matching_files[0]  # 取第一个匹配的文件
            else:
                print(f"未找到以'{name}'开头的txt文件")

            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                class_name = [class_name_prefix+os.path.split(x)[-2] for x in data]  # e.g. 'YoYo' for 'YoYo/v_YoYo_g21_c05'
                data = [x.replace(' ', '_') for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]  # remove extension
                selected_files.extend(data)  # ['v_YoYo_g21_c05', 'v_Typing_g12_c06', ...]
            video_label_dict = dict(zip(selected_files, class_name))  # e.g. 'v_yoyo_g21_c05' -> 'YoYo'
            lists[name] = selected_files
            dict_clsname[name] = video_label_dict

            class_name_unique.extend(list(OrderedDict.fromkeys(class_name)))
        self.train_test_lists = lists
        self.train_test_dict_clsname = dict_clsname  # dict for video label
        self.class_name_unique = class_name_unique

    def get_split(self):
        """ return the current split being used """
        if self.split == "train":
            return self.train_split
        elif self.split == "val":
            return self.val_split
        elif self.split == "test":
            return self.test_split

    def __len__(self):
        """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
        if self.meta_batches:
            return 1000000
        else:
            c = self.get_split()
            return len(c)

    def read_single_image(self, path):
        """Loads a single image from a specified path """
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i

    def frame_sampling(self, n_frames):
        """return a frame id list"""
        if n_frames == self.args.DATA.NUM_INPUT_FRAMES:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.split == "train":
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames - 1 - excess_pad, n_frames - 1)
            else:
                start = 1
                end = n_frames - 2

            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass

            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]

            if self.seq_len == 1:
                idxs = [random.randint(start, end - 1)]
        return idxs

    def load_and_transform_paths(self, paths, query):
        """ loads images from paths and applies transforms. Handles sampling if there are more frames than specified. """
        n_frames = len(paths)
        # idx_f = np.linspace(0, n_frames-1, num=self.args.DATA.NUM_INPUT_FRAMES)   # uniform sampling
        # idxs = [int(f) for f in idx_f]
        idxs = self.frame_sampling(n_frames)
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        transform = self.transform["train"] if self.split == "train" else self.transform["test"]
        video_tensor = [self.tensor_transform(v) for v in imgs]
        video_tensor = torch.stack(video_tensor)  # [T, C, H, W] , have div 255, 0~1
        video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()  # float [C, T, H, W]
        if query and hasattr(self.args.AUGMENTATION, "SUPPORT_QUERY_DIFF") and self.args.AUGMENTATION.SUPPORT_QUERY_DIFF:
            transform = self.transform_query
        imgs = transform(video_tensor).permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
        return imgs

    def load_video_and_transform(self, path, query):
        ''' Load a video from a path and applies transforms'''
        transform = self.transform["train"] if self.split == "train" else self.transform["test"]
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        n_frames = len(vr)
        # idx_f = np.linspace(0, n_frames - 1, num=self.args.DATA.NUM_INPUT_FRAMES)  # uniform sampling
        # idxs = [int(f) for f in idx_f]
        idxs = self.frame_sampling(n_frames)
        vr.seek(0)  # ???
        # frames = vr.get_batch(idxs).asnumpy()  # frames.shape: (n_frames, 240, 320, 3)  <class 'numpy.ndarray'>
        # frames = [torchvision.transforms.ToPILImage()(frame) for frame in frames]
        # video_tensor = [self.tensor_transform(v) for v in frames]

        video_tensor = dlpack.from_dlpack(vr.get_batch(idxs).to_dlpack()).clone()

        # video_tensor = torch.stack(video_tensor)  # [T, C, H, W] , have div 255, 0~1
        # video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()  # float [C, T, H, W]
        if query and hasattr(self.args.AUGMENTATION, "SUPPORT_QUERY_DIFF") and self.args.AUGMENTATION.SUPPORT_QUERY_DIFF:
            transform = self.transform_query
        imgs = transform(video_tensor).permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
        return imgs

    def get_seq(self, label, idx=-1, query=False):
        """Gets a single video sequence for a meta batch.  """
        c = self.get_split()
        if self.meta_batches:
            paths, vid_id = c.get_rand_vid(label, idx)
            if isinstance(paths, list):
                imgs = self.load_and_transform_paths(paths, query)
            else:  # load video files instead of image frames
                imgs = self.load_video_and_transform(paths, query)

            return imgs, vid_id

    def get_meta_batch(self, index):
        """returns dict of support and target images and labels for a meta training task
        support/target labels: 0, 1, 2, 3, 4
        real label: 23, 41, 6, 11, ......
        """
        # select classes to use for this task
        c = self.get_split()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.args.TRAIN.WAY)

        if self.split == "train":
            n_queries = self.args.TRAIN.QUERY_PER_CLASS
        else:
            n_queries = self.args.TRAIN.QUERY_PER_CLASS_TEST

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        support_class_name = []
        target_class_name = []

        for bl, bc in enumerate(batch_classes):
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.TRAIN.SHOT + n_queries)

            for idx in idxs[0:self.args.TRAIN.SHOT]:
                vid, vid_id = self.get_seq(bc, idx, query=False)
                support_set.append(vid)
                support_labels.append(bl)
                real_support_labels.append(bc)
                support_class_name.append(self.class_name_unique[bc])
            for idx in idxs[self.args.TRAIN.SHOT:]:
                vid, vid_id = self.get_seq(bc, idx, query=True)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
                target_class_name.append(self.class_name_unique[bc])

        s = list(zip(support_set, support_labels, real_support_labels, support_class_name))
        random.shuffle(s)
        support_set, support_labels, real_support_labels, support_class_name = zip(*s)

        t = list(zip(target_set, target_labels, real_target_labels, target_class_name))
        random.shuffle(t)
        target_set, target_labels, real_target_labels, target_class_name = zip(*t)

        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes)
        if self.args.VIDEO.HEAD.NAME in ['CNN_OTAM_CLIPFSAR', 'LGA']  and self.split == "test":
            real_support_labels = real_support_labels - self.args.NUM_CLASS
            real_target_labels = real_target_labels - self.args.NUM_CLASS

        return {"support_set": support_set, "support_labels": support_labels, "target_set": target_set,
                "target_labels": target_labels, "real_support_labels": real_support_labels,
                "real_target_labels": real_target_labels, "batch_class_list": batch_classes,
                "support_class_name": support_class_name, "target_class_name": target_class_name}

    def get_single_vid(self, index):
        """gets a single video, used for pretraining the backbone"""
        c = self.get_split()
        paths, gt = c.get_single_video(index)
        vid = self.load_and_transform_paths(paths)

        # return {"images": vid, "target_labels": gt}
        return vid, gt

    def get_all_classname(self):
        return self.class_name_unique

    def __getitem__(self, index):
        if self.meta_batches:
            return self.get_meta_batch(index)
        else:
            return self.get_single_vid(index)
