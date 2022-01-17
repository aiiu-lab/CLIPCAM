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

from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.dataset import ImageNetDataset
from utils.imagenet_utils import *
from utils.evaluation_tools import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='/scratch2/users/jtchen0528/Datasets/ImageNet/validation',
                    help="directory of ImageNet dataset")
parser.add_argument("--save_dir", type=str, default='eval_result',
                    help="directory to save the result")
parser.add_argument("--gpu_id", type=int, default=1,
                    help="GPU to work on")
parser.add_argument("--batch", type=int, default=32,
                    help="batch size")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--distill_num", type=int, default=0,
                    help="Number of iterative masking")
parser.add_argument("--mask_threshold", type=float, default=0.2,
                    help="Threshold of the mask")
parser.add_argument("--attack", type=str, default='None',
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--sentence_prefix", type=str, default='word',
                    help="input text type: \"sentence\", \"word\"")
parser.add_argument("--save_result", type=int, default=0,
                    help="save result or not")
args = parser.parse_args()

DATA_DIR = args.data_dir
SAVE_DIR = args.save_dir
SENTENCE_PREFIX = args.sentence_prefix
GPU_ID = args.gpu_id
BATCH_SIZE = args.batch
CLIP_MODEL_NAME = args.clip_model_name
CAM_MODEL_NAME = args.cam_model_name
RESIZE = args.resize
DISTILL_NUM = args.distill_num
MASK_THRESHOLD = args.mask_threshold
ATTACK = args.attack
SAVE_RESULT = args.save_result
if CLIP_MODEL_NAME.split('-')[-1] == 'pretrained':
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

dataset = ImageNetDataset(data_dir=DATA_DIR, transform=ImageTransform, original_transform=originalTransform, gpu_id=GPU_ID, attack=ATTACK)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

if not PRETRAINED:
    zeroshot_weights, class_sentences, class_words = zeroshot_classifier(imagenet_classes, imagenet_templates, model, GPU_ID)

top1, top5, loc_top1, loc_top5, n = 0., 0., 0., 0., 0.
count = 0
for i, (images, targets, gt_masks, orig_image) in enumerate(tqdm(loader)):
    images = images.to(GPU_ID)
    targets = targets.to(GPU_ID)
    gt_masks = gt_masks.to(GPU_ID)
    orig_image = orig_image.to(GPU_ID)
    image_paths = dataset.data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]    
    image_names = [p[1].split('/')[-1].split('.')[0] for p in image_paths]

    # predict
    if not PRETRAINED:
        with torch.no_grad():
            image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights
    else:
        with torch.no_grad():
            output = model(images)
        logits = torch.nn.functional.softmax(output, dim=-1)


    # measure accuracy
    cls_acc, loc_acc = accuracy(logits, targets, topk=(1, 5))
    acc1, acc5 = cls_acc
    correct_indice_top1, correct_indice_topk = loc_acc

    images_cam = torch.index_select(images, 0, correct_indice_topk)
    targets_cam = torch.index_select(targets, 0, correct_indice_topk)
    gt_masks_cam = torch.index_select(gt_masks, 0, correct_indice_topk)
    orig_image_cam = torch.index_select(orig_image, 0, correct_indice_topk)
    image_names_cam = [image_names[i.item()] for i in correct_indice_topk]

    if not PRETRAINED:
        if SENTENCE_PREFIX == 'sentence':
            sentence_features_cam = torch.index_select(class_sentences, 0, targets_cam)
        elif SENTENCE_PREFIX == 'word':
            sentence_features_cam = torch.index_select(class_words, 0, targets_cam)
        grayscale_cam = cam(input_tensor=images_cam, text_tensor=sentence_features_cam)
    else:
        grayscale_cam = cam(input_tensor=images_cam, target_category=targets_cam)

    grayscale_cam_mask = np.where(grayscale_cam < MASK_THRESHOLD, 0, 1)

    pred_bbox, pred_mask = MaskToBBox(grayscale_cam_mask, images_cam.size(0))
    grayscale_cam_tensor = torch.from_numpy(pred_mask).to(GPU_ID)
    ious = iou_pytorch(outputs=grayscale_cam_tensor, labels=gt_masks_cam)

    loc_acc1_ious = torch.index_select(ious, 0, correct_indice_top1)

    loc_acc5 = (ious >= 0.5).sum()
    loc_acc1 = (loc_acc1_ious >= 0.5).sum()

    top1 += acc1
    top5 += acc5
    loc_top1 += loc_acc1
    loc_top5 += loc_acc5
    n += images.size(0)

    if SAVE_RESULT:
        gt_bboxes, gt_masks = MaskToBBox(gt_masks_cam.cpu().numpy(),  images_cam.size(0))
        for mask_num in range(len(grayscale_cam)):
            label = imagenet_labels[targets_cam[mask_num].item()]
            os.makedirs(os.path.join(SAVE_DIR, label), exist_ok=True)
            getHeatMap(grayscale_cam[mask_num], orig_image_cam[mask_num].permute(1, 2, 0).cpu().numpy(), os.path.join(SAVE_DIR, label, image_names_cam[mask_num] + '.png'), pred_bbox[mask_num], gt_bboxes[mask_num])

    if i  % 50 == 0:
        print(f"Done {((i + 1) / len(loader) * 100)}%")

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100 
loc_top1 = (loc_top1 / n) * 100
loc_top5 = (loc_top5 / n) * 100 

print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")
print(f"Top-1 localization accuracy: {loc_top1:.2f}")
print(f"Top-5 localization accuracy: {loc_top5:.2f}")

with open(os.path.join(SAVE_DIR, 'result.txt'), 'w') as f:
    f.write(f"Top1 Accuracy: {top1:.2f}\nTop5 Accuracy: {top5:.2f}\nTop1 Localization Accuracy: {loc_top1:.2f}\nTop5 Localization Accuracy: {loc_top5:.2f}")