from PIL import Image
import torch
from torch.utils import data
import os
import numpy as np
import xml.etree.ElementTree as ET
from torchvision import transforms
from utils.imagenet_utils import *
from utils.preprocess import getAttacker
import pandas as pd
from utils.Parser import HICOparser
from tqdm import tqdm
import random

class ImageNetDataset(data.Dataset):
    def __init__(self, data_dir, transform, original_transform, gpu_id = 0, attack = None):
        self.dir = data_dir

        data_list = []
        for classes in os.listdir(self.dir):
            
            curr_class_name = imagenet_classes[int(imagenet_labels.index(classes))]
            # if curr_class_name in ['vespa', 'grasshopper', 'otter', 'sea lion', 'French horn']:
            for pic in os.listdir(os.path.join(self.dir, classes)):
                path = os.path.join(self.dir, classes, pic).replace('\\', '/')
                data_list.append([classes, path])

        self.data_list = data_list
        self.transform = transform
        self.original_transform = original_transform
        self.attack = attack
        self.gpu_id = gpu_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index][1]
        image = Image.open(image_path)
        if self.attack is not None:
            image = image.resize((224, 224))
            image = getAttacker(image, type=self.attack, gpu_id=self.gpu_id)
        orig_image = self.original_transform(image)
        class_id = torch.from_numpy(np.array(self.searchClassIndex(self.data_list[index][0])))
        image = self.transform(image)
        c, h, w = image.size()
        gt_bbox_mask = self.getGTMask(image_path.split('/')[-1], w, h)
        return image, class_id, gt_bbox_mask, orig_image

    def searchClassName(self, label):
        return [imagenet_classes[i] for i in range(1000) if imagenet_labels[i] == label][0]

    def searchClassIndex(self, label):
        return [i for i in range(1000) if imagenet_labels[i] == label][0]

    def getGTMask(self, pic_name, w, h):
        annotation = ET.parse(os.path.join(self.dir, '..', 'bbox', 'val', pic_name.split('.')[0] + '.xml'))
        annotation_root = annotation.getroot()
        orig_w, orig_h = int(annotation_root[3][0].text), int(annotation_root[3][1].text)
        gt_bboxes = []
        for bbox in annotation_root[5:]:
            gt_bboxes.append([int(bbox[4][0].text), int(bbox[4][1].text), int(bbox[4][2].text), int(bbox[4][3].text)])
        gt_mask = np.zeros((w, h)).astype('uint8')
        for bbox in gt_bboxes:
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            xmin, ymin, xmax, ymax = int(xmin / orig_w * w), int(ymin / orig_h * h), int(xmax / orig_w * w), int(ymax / orig_h * h)
            gt_mask[ymin: ymax+1, xmin:xmax+1] = 1

        filtered_gt_mask_cordinates = np.array(np.where(gt_mask==1))
        gt_total_mask_bbox = np.array((np.min(filtered_gt_mask_cordinates[1]), np.min(filtered_gt_mask_cordinates[0]), np.max(filtered_gt_mask_cordinates[1]), np.max(filtered_gt_mask_cordinates[0])))
        gt_total_mask = np.zeros((w, h)).astype('uint8')
        xmin, ymin, xmax, ymax = gt_total_mask_bbox[0], gt_total_mask_bbox[1], gt_total_mask_bbox[2], gt_total_mask_bbox[3]
        gt_total_mask[ymin: ymax+1, xmin:xmax+1] = 1

        return gt_total_mask


class DirDataset(data.Dataset):
    def __init__(self, data_dir, transform, original_transform):
        self.dir = data_dir

        data_list = []
        for pic in os.listdir(os.path.join(self.dir)):
            path = os.path.join(self.dir, pic).replace('\\', '/')
            data_list.append(path)

        self.data_list = data_list
        self.transform = transform
        self.original_transform = original_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path)
        orig_image = self.original_transform(image)
        image = self.transform(image)
        return image, orig_image

class OpenImageDataset(data.Dataset):
    def __init__(self, data_dir, transform, original_transform, gpu_id = 0, attack = None):
        self.dir = data_dir

        self.detection_data = pd.read_csv(os.path.join(self.dir, '..', 'labels', 'detections.csv'))
        self.class_names = pd.read_csv(os.path.join(self.dir, '..', 'metadata', 'classes.csv'), names=['ID', 'Class'], header=None)

        data_list = []
        for pic in os.listdir(self.dir):
            path = os.path.join(self.dir, pic).replace('\\', '/')
            data_list.append(path)

        self.data_list = data_list
        self.transform = transform
        self.original_transform = original_transform
        self.attack = attack
        self.gpu_id = gpu_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path)
        if self.attack is not None:
            image = image.resize((224, 224))
            image = getAttacker(image, type=self.attack, gpu_id=self.gpu_id)
        orig_image = self.original_transform(image)
        image = self.transform(image)

        return image, orig_image

    def searchClassName(self, label):
        return self.class_names.loc[self.class_names['ID'] == label]['Class'].tolist()[0].lower()

    def searchClassNameFromID(self, index):
        return self.class_names['Class'][index].lower()

    def searchClassIndex(self, label):
        return self.class_names.loc[self.class_names['ID'] == label].index.tolist()[0]

    def getGTMasks(self, pic_name, w, h):
        annotations = self.detection_data.loc[self.detection_data['ImageID'] == pic_name]
        labels = annotations['LabelName'].unique().tolist()
        label_indices = [self.searchClassIndex(label) for label in labels]
        gt_total_masks = []
        for label in labels:
            masks_pd = annotations.loc[annotations['LabelName'] == label]
            mask_total = np.zeros((w, h))
            for index, mask in masks_pd.iterrows():
                # mask_im = Image.open(os.path.join(self.dir, '..', 'labels', 'masks', mask['MaskPath'][0].upper(), mask['MaskPath'])).convert('L')
                xmin, xmax, ymin, ymax = int(mask['XMin'] * w),  int(mask['XMax'] * w),  int(mask['YMin'] * h),  int(mask['YMax'] * h)
                mask_total[ymin: ymax+1, xmin:xmax+1] = 1
            #     mask_total += np.array(mask_im)
            # mask_total = np.where(mask_total != 0, 1, 0)

            filtered_gt_mask_cordinates = np.array(np.where(mask_total==1))
            gt_total_mask_bbox = np.array((np.min(filtered_gt_mask_cordinates[1]), np.min(filtered_gt_mask_cordinates[0]), np.max(filtered_gt_mask_cordinates[1]), np.max(filtered_gt_mask_cordinates[0])))
            gt_total_mask = np.zeros((w, h)).astype('uint8')
            xmin, ymin, xmax, ymax = gt_total_mask_bbox[0], gt_total_mask_bbox[1], gt_total_mask_bbox[2], gt_total_mask_bbox[3]
            gt_total_mask[ymin: ymax+1, xmin:xmax+1] = 1
            gt_total_masks.append(gt_total_mask)
        label_indices = np.asarray(label_indices)
        gt_total_masks = np.asarray(gt_total_masks)

        return label_indices, gt_total_masks

HICO_filtered_actions = ['sipping', 'loading', 'lying on', 'sitting on', 'jumping', 'checking', 'kissing', 'sailing', 'directing', 'smelling', 'feeding', 'catching', 'assembling', 'greeting', 'herding', 'hitting', 'licking', 'grooming', 'drinking with', 'flipping', 'texting on', 'cleaning', 'holding', 'hosing', 'pulling', 'scratching', 'teaching', 'flushing', 'tagging', 'typing on', 'training', 'stirring', 'milking', 'carrying', 'shearing', 'operating', 'pouring', 'peeling', 'swinging', 'cutting', 'cutting with', 'rowing', 'standing under', 'breaking', 'throwing', 'reading', 'spining', 'buying', 'racing', 'filling', 'wearing', 'controling']

class HICODataset(data.Dataset):
    def __init__(self, data_dir, transform, original_transform, split='train', mode='full'):
        self.dir = data_dir
        parser = HICOparser(self.dir)

        if split == 'train':
            self.feat_with_image_names = parser.getTrainFeatswithImageNames()
        elif split == 'test':
            self.feat_with_image_names = parser.getTestFeatswithImageNames()
        index2hoi = parser.getIndex2HOI()

        raw_data = []
        gt_action = []
        gt_object = []
        pbar = tqdm(self.feat_with_image_names.items())
        for image_path, feat in pbar:
            image_name = image_path.split('/')[-1].split('.')[0]

            # get the image height, width and the bboxs corresponded with the hoi
            img_height = feat['sizes'][0]
            img_width = feat['sizes'][1]
            use_rgb = feat['sizes'][2] == 3
            bboxes_human = feat['bbox_human']
            bboxes_object = feat['bbox_object']
            gt_ids = feat['ground_truth_ids'][0]

            if len(bboxes_object) != 1 or len(bboxes_human) != 1:
                continue

            for i in range(len(bboxes_object)):
                if not (index2hoi[gt_ids][1] == 'no interaction'):
                    if mode != 'half':
                        raw_data.append([image_path, img_height, img_width, bboxes_human, bboxes_object, gt_ids, index2hoi[gt_ids][1]])
                        gt_action.append(index2hoi[gt_ids][1])
                        gt_object.append(index2hoi[gt_ids][0])
                    else:
                        if index2hoi[gt_ids][1] in HICO_filtered_actions:
                            raw_data.append([image_path, img_height, img_width, bboxes_human, bboxes_object, gt_ids, index2hoi[gt_ids][1]])
                            gt_action.append(index2hoi[gt_ids][1])
                            gt_object.append(index2hoi[gt_ids][0])
                    
        
        self.gt_actions = sorted(list(set(gt_action)))
        self.gt_objects = sorted(list(set(gt_object)))
        
        if mode == 'few':
            most_data = [d for d in raw_data if d[-1] in HICO_filtered_actions]
            filtered_data = [d for d in raw_data if d[-1] not in HICO_filtered_actions]
            filtered_data_actions = [item for item in self.gt_actions if item not in HICO_filtered_actions]
            few_data = []
            for action in filtered_data_actions:
                filtered_action = [d for d in filtered_data if d[-1] == action]
                if len(filtered_action) >= 4:
                    few_data += random.sample(filtered_action, 4)
                else:
                    few_data += filtered_action
            raw_data = most_data + few_data
            
        self.data_list = raw_data
        self.index2hoi = index2hoi
        self.transform = transform
        self.original_transform = original_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data[0]
        img_height = data[1]
        img_width = data[2]
        bboxes_human = data[3]
        bboxes_object = data[4]
        gt_ids = data[5]
        image = Image.open(image_path)

        image = self.getImage(image_path)
        mask_object = self.bboxs2Mask(bboxes_object[0], img_height, img_width)
        mask_human = self.bboxs2Mask(bboxes_human[0], img_height, img_width)
        mask = (mask_human | mask_object)
        # mask_im = Image.fromarray(
        #     (mask * 255).astype(np.uint8)).convert('L')

        orig_image = self.original_transform(image)
        image = self.transform(image)
        return image, orig_image, mask, gt_ids
    
    def getImage(self, image_id):
        image_path = os.path.join(image_id)
        image = Image.open(image_path).convert("RGB")
        return image

    def bboxs2Mask(self, bbx, h, w):
        mask = np.zeros((224, 224)).astype('uint8')
        xmin, ymin, xmax, ymax = bbx
        xmin, xmax, ymin, ymax = int(
            xmin * 224.0 / w), int(xmax * 224.0 / w), int(ymin * 224.0 / h),  int(ymax * 224.0 / h)
        mask[ymin: ymax+1, xmin:xmax+1] = 1
        return mask
