#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :wxf_dataset.py
@Description  :
@Date         :2021/12/24 16:34:35
@Author       :Arctic Little Pig
@version      :1.0
'''

import os

import numpy as np
import torch
import torch.nn.functional as F
import ujson
from PIL import Image
from torch.serialization import save
from torch.utils.data import Dataset

from dataloaders.data_utils import get_unk_mask_indices

category_info = {'Dawn': 0, 'Morning': 1, 'Afternoon': 2, 'Dusk': 3,
                 'Cloudy': 4, 'Sunny': 5, 'Rainy': 6}

idx_category = {0: 'Dawn', 1: 'Morning', 2: 'Afternoon', 3: 'Dusk',
                4: 'Cloudy', 5: 'Sunny', 6: 'Rainy'}


class WxfDataset(Dataset):
    def __init__(self, img_dir='./data/wxf/train_dataset/', anno_path='./data/wxf/train_dataset/train.json', indices=None, image_transform=None, known_labels=0, testing=False, use_difficult=False):
        self.img_dir = img_dir
        self.predict = False if anno_path else True
        if not self.predict:
            with open(anno_path, 'r') as f:
                anno = ujson.load(f)
            self.anno = anno["annotations"]

        self.num_labels = 7
        self.known_labels = known_labels
        self.testing = testing

        if not self.predict:
            self.img_names = []
            self.labels = []
            for annotation in self.anno:
                img_name = annotation["filename"].split("\\")[-1]
                self.img_names.append(img_name)

                label_period = annotation["period"]
                label_weather = annotation["weather"]

                label_vector = np.zeros(self.num_labels)
                label_vector[int(category_info[label_period])] = 1.0
                label_vector[int(category_info[label_weather])] = 1.0
                self.labels.append(label_vector)
        else:
            self.img_names = os.listdir(self.img_dir)
            self.labels = np.zeros((len(self.img_names), self.num_labels))

        # self.labels = np.array(self.labels).astype(np.float32)
        self.labels = np.array(self.labels).astype(int)

        if indices is not None:
            self.img_names = np.array(self.img_names)
            self.img_names = self.img_names[indices].tolist()
            self.labels = self.labels[indices]

        self.image_transform = image_transform
        self.epoch = 1

    def __getitem__(self, index):
        name = self.img_names[index]
        image = Image.open(os.path.join(self.img_dir, name)).convert('RGB')

        if self.image_transform:
            # 将image转化为torch.tensor
            image = self.image_transform(image)

        labels = torch.Tensor(self.labels[index])

        unk_mask_indices = get_unk_mask_indices(
            image, self.testing, self.num_labels, self.known_labels, self.epoch)

        mask = labels.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(name)

        return sample

    def __len__(self):
        return len(self.img_names)

    def convertToLabel(self, predictions, img_ids, save_path):
        assert len(predictions) == self.__len__(), \
            "the size of predictions is not consistent with the size of dataset"

        optimal_threshold = 0.5

        predictions = F.sigmoid(predictions)
        predictions = predictions.numpy()
        top_2nd = np.sort(predictions)[:, -2:].tolist()
        # predictions_top2 = predictions.copy()
        # predictions_top2[predictions_top2 < top_2nd] = 0
        # predictions_top2[predictions_top2 >= top_2nd] = 1
        top_2idx = np.argsort(predictions)[:, -2:]

        output = dict()
        output["annotations"] = []

        for i in range(self.__len__()):
            predict_dict = dict()
            predict_dict["filename"] = "test_images\\" + img_ids[i]
            predict_dict["class"] = []
            cate_idx = top_2idx[i]
            cate_confidence = top_2nd[i]
            for j, cate in enumerate(cate_idx):
                class_dict = dict()
                if cate < 4:
                    class_dict["period"] = idx_category[cate]
                else:
                    class_dict["weather"] = idx_category[cate]
                class_dict["confidence"] = cate_confidence[j]
                predict_dict["class"].append(class_dict)
            output["annotations"].append(predict_dict)

        output_path = os.path.join(save_path, "test_predict.json")
        with open(output_path, "w") as fp:
            ujson.dump(output, fp)
