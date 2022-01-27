import clip_modified
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import patches as mtp_ptch
from torchvision import transforms
from tqdm.notebook import tqdm
import argparse
import cv2
import pandas as pd
import json

from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.dataset import DirDataset
from utils.imagenet_utils import *
from utils.evaluation_tools import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='datasets/OpenImage/validation/data',
                    help="directory of dataset")
parser.add_argument("--dataset", type=str, default='',
                    help="dataset name")
parser.add_argument("--save_dir", type=str, default='eval_result/rn50-grad',
                    help="directory to save the result")
parser.add_argument("--gpu_id", type=int, default=0,
                    help="GPU to work on")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--distill_num", type=int, default=0,
                    help="Number of iterative masking")
parser.add_argument("--attack", type=str, default=None,
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--sentence_prefix", type=str, default='',
                    help="input text type: \"sentence\", \"word\" (only for OpenImage)")
parser.add_argument("--mask_threshold", type=float, default=0.2,
                    help="Threshold of the localization mask")
args = parser.parse_args()

DATA_DIR = args.data_dir
DATASET_NAME = args.dataset
SAVE_DIR = args.save_dir
SENTENCE_PREFIX = args.sentence_prefix
GPU_ID = args.gpu_id
BATCH_SIZE = 1
CLIP_MODEL_NAME = args.clip_model_name
CAM_MODEL_NAME = args.cam_model_name
RESIZE = args.resize
ATTACK = args.attack
DISTILL_NUM = args.distill_num
MASK_THRESHOLD = args.mask_threshold
if CLIP_MODEL_NAME == 'RN50-pretrained' or CLIP_MODEL_NAME == 'ViT-pretrained':
    PRETRAINED = True
else:
    PRETRAINED = False

os.makedirs(SAVE_DIR, exist_ok=True)

model, target_layer, reshape_transform = getCLIP(
    model_name=CLIP_MODEL_NAME, gpu_id=GPU_ID)

cam = getCAM(model_name=CAM_MODEL_NAME, model=model, target_layer=target_layer,
             gpu_id=GPU_ID, reshape_transform=reshape_transform)

ImageTransform = getImageTranform(resize=RESIZE)
originalTransform = getImageTranform(resize=RESIZE, normalized=False)

dataset = DirDataset(data_dir=DATA_DIR, transform=ImageTransform, original_transform=originalTransform, gpu_id=GPU_ID, attack=ATTACK)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

if DATASET_NAME == 'openimage':
    filepath = os.path.join(DATA_DIR, '..', 'labels', 'detections.csv')
    segmentation_data = pd.read_csv(filepath)
    class_name = pd.read_csv(os.path.join(DATA_DIR, '..', 'metadata', 'classes.csv'), names=['ID', 'Class'], header=None)
    def getNamebyLabel(label):
        return class_name.loc[class_name['ID'] == label]['Class'].tolist()[0]
    def getOpenImageAnnotation(paths):
        for path in paths:
            item_info = segmentation_data.loc[segmentation_data['ImageID'] == path.split('/')[-1].split('.')[0]]
            labels = item_info['LabelName'].unique().tolist()
            labels = [getNamebyLabel(label).lower() for label in labels]
            sentences = [f'{SENTENCE_PREFIX + label}' for label in labels]
        return sentences, labels
elif DATASET_NAME == 'cosmos':
    with open(os.path.join(DATA_DIR, '..', 'annotations', 'test_data.json')) as f:
        article_list = [json.loads(line) for line in f]
    test_samples = article_list
    def getCosmosAnnotation(paths):
        for path in paths:
            item_info = test_samples[int(path.split('/')[-1].split('.')[-2])]
            sentences = [item_info['caption1_modified'], item_info['caption2_modified']]
            labels = ['1', '2']
        return sentences, labels


for i, (images, orig_image) in enumerate(tqdm(loader)):
    images = images.to(GPU_ID)
    orig_image = orig_image.to(GPU_ID)
    image_paths = dataset.data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    orig_image = orig_image.squeeze(0)
    if DATASET_NAME == 'openimage':
        sentences, labels = getOpenImageAnnotation(image_paths)
    elif DATASET_NAME == 'cosmos':
        sentences, labels = getCosmosAnnotation(image_paths)
    else:
        sentences, labels = None, None
    
    if sentences == None:
        sentences = [input(f"Please enter the query sentence for {image_paths[0]}: ")]
        labels = ['']

    for (sentence, label) in zip(sentences, labels):
        # with torch.no_grad():
        text  = clip_modified.tokenize(sentence)
        text = text.to(GPU_ID)
        text_features = model.encode_text(text)
        grayscale_cam = cam(input_tensor=images, text_tensor=text_features)[0, :]
        grayscale_cam_total = [grayscale_cam]

        for distill in range(DISTILL_NUM):
            distill_mask = np.where(grayscale_cam > 0.5, 0, 1)
            distill_mask = torch.tensor(distill_mask).unsqueeze(0)
            distill_mask = torch.cat((distill_mask,distill_mask,distill_mask)).unsqueeze(0)
            distill_mask = distill_mask.to(GPU_ID)
            images = images * distill_mask
            grayscale_cam = cam(input_tensor=images, text_tensor=text_features)[0, :]
            grayscale_cam_total += grayscale_cam
        if DISTILL_NUM > 0:
            grayscale_cam = (grayscale_cam_total / np.max(grayscale_cam_total))[0, :]

        getHeatMapNoBBox(grayscale_cam, orig_image.permute(1, 2, 0).cpu().numpy(), os.path.join(SAVE_DIR, str(image_paths[0].split('/')[-1].split('.')[0] + '_' + label) + '.png'))

    if i % 50 == 0:
        print(f"Done {((i + 1) / len(loader) * 100)}%")