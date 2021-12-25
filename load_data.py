#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :load_data.py
@Description  :
@Date         :2021/12/22 16:49:39
@Author       :Arctic Little Pig
@version      :1.0
'''

import os
import warnings

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

from dataloaders.coco80_dataset import Coco80Dataset
from dataloaders.coco1000_dataset import Coco1000Dataset
from dataloaders.cub312_dataset import CUBDataset
from dataloaders.news500_dataset import NewsDataset
from dataloaders.vg500_dataset import VGDataset
from dataloaders.voc2007_20 import Voc07Dataset
from dataloaders.wxf_dataset import WxfDataset

# warnings.filterwarnings("ignore")


def get_data(args):
    dataset = args.dataset
    data_root = args.dataroot
    batch_size = args.batch_size

    rescale = args.scale_size
    random_crop = args.crop_size

    workers = args.workers
    # for cub dataset
    attr_group_dict = args.attr_group_dict
    n_groups = args.n_groups

    # 从imageNet数据集中得到的均值和标准差
    normTransform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = rescale
    crop_size = random_crop
    if args.test_batch_size == -1:
        args.test_batch_size = batch_size

    trainTransform = transforms.Compose([
        # 将输入图像resize到 640*640
        transforms.Resize((scale_size, scale_size)),
        # 对resize后的图像进行随机裁剪至 rand_size*rand_size
        transforms.RandomChoice([
            transforms.RandomCrop(640),
            transforms.RandomCrop(576),
            transforms.RandomCrop(512),
            transforms.RandomCrop(384),
            transforms.RandomCrop(320)
        ]),
        # 将rand_crop后的图像resize至 576*576
        transforms.Resize((crop_size, crop_size)),
        # 依概率将resize后的图像进行水平翻转，默认翻转的概率是0.5
        transforms.RandomHorizontalFlip(),
        # 将图像的RGB（0~255）转化为灰度（0~1）
        transforms.ToTensor(),
        # 对灰度值图像进行归一化
        normTransform])

    testTransform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        # 对resize后的图像进行中心裁剪至 crop_size*crop_size
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normTransform])

    test_dataset = None
    test_loader = None
    predict_dataset = None
    predict_loader = None
    drop_last = False
    if dataset == 'coco':
        coco_root = os.path.join(data_root, 'coco')
        ann_dir = os.path.join(coco_root, 'annotations_pytorch')
        train_img_root = os.path.join(coco_root, 'train2014')
        test_img_root = os.path.join(coco_root, 'val2014')
        train_data_name = 'train.data'
        val_data_name = 'val_test.data'

        train_dataset = Coco80Dataset(
            split='train',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root, train_data_name),
            img_root=train_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=trainTransform,
            known_labels=args.train_known_labels,
            testing=False)
        valid_dataset = Coco80Dataset(split='val',
                                      num_labels=args.num_labels,
                                      data_file=os.path.join(
                                          coco_root, val_data_name),
                                      img_root=test_img_root,
                                      annotation_dir=ann_dir,
                                      max_samples=args.max_samples,
                                      transform=testTransform,
                                      known_labels=args.test_known_labels,
                                      testing=True)

    elif dataset == 'coco1000':
        ann_dir = os.path.join(data_root, 'coco', 'annotations_pytorch')
        data_dir = os.path.join(data_root, 'coco')
        train_img_root = os.path.join(data_dir, 'train2014')
        test_img_root = os.path.join(data_dir, 'val2014')

        train_dataset = Coco1000Dataset(
            ann_dir, data_dir, split='train', transform=trainTransform, known_labels=args.train_known_labels, testing=False)
        valid_dataset = Coco1000Dataset(
            ann_dir, data_dir, split='val', transform=testTransform, known_labels=args.test_known_labels, testing=True)

    elif dataset == 'vg':
        vg_root = os.path.join(data_root, 'VG')
        train_dir = os.path.join(vg_root, 'VG_100K')
        train_list = os.path.join(vg_root, 'train_list_500.txt')
        test_dir = os.path.join(vg_root, 'VG_100K')
        test_list = os.path.join(vg_root, 'test_list_500.txt')
        train_label = os.path.join(
            vg_root, 'vg_category_500_labels_index.json')
        test_label = os.path.join(vg_root, 'vg_category_500_labels_index.json')

        train_dataset = VGDataset(
            train_dir,
            train_list,
            trainTransform,
            train_label,
            known_labels=0,
            testing=False)
        valid_dataset = VGDataset(
            test_dir,
            test_list,
            testTransform,
            test_label,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'news':
        drop_last = True
        ann_dir = '/bigtemp/jjl5sw/PartialMLC/data/bbc_data/'

        train_dataset = NewsDataset(
            ann_dir, split='train', transform=trainTransform, known_labels=0, testing=False)
        valid_dataset = NewsDataset(
            ann_dir, split='test', transform=testTransform, known_labels=args.test_known_labels, testing=True)

    elif dataset == 'voc':
        voc_root = os.path.join(data_root, 'voc/VOCdevkit/VOC2007/')
        img_dir = os.path.join(voc_root, 'JPEGImages')
        anno_dir = os.path.join(voc_root, 'Annotations')
        train_anno_path = os.path.join(voc_root, 'ImageSets/Main/trainval.txt')
        test_anno_path = os.path.join(voc_root, 'ImageSets/Main/test.txt')

        train_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=train_anno_path,
            image_transform=trainTransform,
            labels_path=anno_dir,
            known_labels=args.train_known_labels,
            testing=False,
            use_difficult=False)
        valid_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=test_anno_path,
            image_transform=testTransform,
            labels_path=anno_dir,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'wxf':
        wxf_root = os.path.join(data_root, 'wxf')
        train_img_dir = os.path.join(wxf_root, 'train_dataset/train_images')
        data_size = len(os.listdir(train_img_dir))
        train_size = int(data_size * args.train_ratio)
        train_indices = np.random.choice(data_size, train_size, replace=False)
        valid_indices = np.delete(np.arange(data_size), train_indices)

        train_anno_path = os.path.join(wxf_root, 'train_dataset/train.json')
        predict_img_dir = os.path.join(wxf_root, 'test_dataset/test_images')

        train_dataset = WxfDataset(
            img_dir=train_img_dir,
            anno_path=train_anno_path,
            indices=train_indices,
            image_transform=trainTransform,
            known_labels=args.train_known_labels,
            testing=False,
            use_difficult=False)
        valid_dataset = WxfDataset(
            img_dir=train_img_dir,
            anno_path=train_anno_path,
            indices=valid_indices,
            image_transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)
        predict_dataset = WxfDataset(
            img_dir=predict_img_dir,
            anno_path=None,
            indices=None,
            image_transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'cub':
        drop_last = True
        resol = 299
        resized_resol = int(resol * 256/224)

        trainTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            # transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

        testTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

        cub_root = os.path.join(data_root, 'CUB_200_2011')
        image_dir = os.path.join(cub_root, 'images')
        train_list = os.path.join(
            cub_root, 'class_attr_data_10', 'train_valid.pkl')
        valid_list = os.path.join(
            cub_root, 'class_attr_data_10', 'train_valid.pkl')
        test_list = os.path.join(cub_root, 'class_attr_data_10', 'test.pkl')

        train_dataset = CUBDataset(image_dir, train_list, trainTransform, known_labels=args.train_known_labels,
                                   attr_group_dict=attr_group_dict, testing=False, n_groups=n_groups)
        valid_dataset = CUBDataset(image_dir, valid_list, testTransform, known_labels=args.test_known_labels,
                                   attr_group_dict=attr_group_dict, testing=True, n_groups=n_groups)
        test_dataset = CUBDataset(image_dir, test_list, testTransform, known_labels=args.test_known_labels,
                                  attr_group_dict=attr_group_dict, testing=True, n_groups=n_groups)

    else:
        print('no dataset available')
        exit(0)

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers, drop_last=drop_last)
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)
    if predict_dataset is not None:
        predict_loader = DataLoader(
            predict_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)

    return train_loader, valid_loader, test_loader, predict_loader
