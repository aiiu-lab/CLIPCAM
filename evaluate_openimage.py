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
from utils.dataset import OpenImageDataset
from utils.imagenet_utils import *
from utils.evaluation_tools import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='/scratch2/users/jtchen0528/Datasets/OpenImage/test/data',
                    help="directory of OpenImage")
parser.add_argument("--save_dir", type=str, default='eval_result',
                    help="directory to save the result")
parser.add_argument("--gpu_id", type=int, default=1,
                    help="GPU to run on")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not, 1 or 0")
parser.add_argument("--distill_num", type=int, default=0,
                    help="Number of iterative masking")
parser.add_argument("--mask_threshold", type=float, default=0.2,
                    help="Threshold of the localization mask")
parser.add_argument("--attack", type=str, default='None',
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--sentence_prefix", type=str, default='',
                    help="Text input prefix: \"PREFIX\" + \"object class name\"")
parser.add_argument("--save_result", type=int, default=0,
                    help="save result or not, 1 or 0")
args = parser.parse_args()

DATA_DIR = args.data_dir
SAVE_DIR = args.save_dir
SENTENCE_PREFIX = args.sentence_prefix
GPU_ID = args.gpu_id
BATCH_SIZE = 1
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

print(CLIP_MODEL_NAME, CAM_MODEL_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)

model, target_layer, reshape_transform = getCLIP(
    model_name=CLIP_MODEL_NAME, gpu_id=GPU_ID)

cam = getCAM(model_name=CAM_MODEL_NAME, model=model, target_layer=target_layer,
             gpu_id=GPU_ID, reshape_transform=reshape_transform)

ImageTransform = getImageTranform(resize=RESIZE)
originalTransform = getImageTranform(resize=RESIZE, normalized=False)

dataset = OpenImageDataset(data_dir=DATA_DIR, transform=ImageTransform, original_transform=originalTransform, gpu_id=GPU_ID, attack=ATTACK)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# if not PRETRAINED:
#     zeroshot_weights, class_sentences, class_words = zeroshot_classifier(imagenet_classes, imagenet_templates, model, GPU_ID)

Final_result = []

total_acc, loc_acc, n = 0., 0., 0.
for i, (images, orig_image) in enumerate(tqdm(loader)):
    images = images.to(GPU_ID)
    orig_image = orig_image.to(GPU_ID)
    image_paths = dataset.data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    image_names = [p.split('/')[-1].split('.')[0] for p in image_paths]

    images_embeddings = []
    orig_images_embeddings = []
    gt_mask_total = []
    text_embeddings = []
    label_names = []
    image_names_total = []
    for image_index, image in enumerate(images):
        c, h, w = image.size()
        label_indices, gt_bbox_masks = dataset.getGTMasks(image_paths[image_index].split('/')[-1].split('.')[0], w, h)
        label_indices = torch.from_numpy(label_indices).to(GPU_ID)
        gt_bbox_masks = torch.from_numpy(gt_bbox_masks).to(GPU_ID)
        for label_index, label in enumerate(label_indices):
            label_name = dataset.searchClassNameFromID(label.item())
            sentence = [f'{SENTENCE_PREFIX}{label_name}']
            label_names.append(label_name)
            if not PRETRAINED:
                with torch.no_grad():
                    text_token = clip_modified.tokenize(sentence).to(GPU_ID) #tokenize
                    text_embedding = model.encode_text(text_token) #embed with text encoder
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                text_embedding = text_embedding.mean(dim=0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)
            images_embeddings.append(image)
            orig_images_embeddings.append(orig_image[image_index])
            gt_mask_total.append(gt_bbox_masks[label_index])
            image_names_total.append(image_names[image_index])
    if len(images_embeddings) !=  0:
        images_embeddings = torch.stack(images_embeddings, dim=0).to(GPU_ID)
        orig_images_embeddings = torch.stack(orig_images_embeddings, dim=0).to(GPU_ID)
        gt_mask_total = torch.stack(gt_mask_total, dim=0).to(GPU_ID)
        if not PRETRAINED:
            text_embeddings = torch.stack(text_embeddings, dim=0).to(GPU_ID)

        first = True
        inner_batch_index = -1
        for inner_batch_index in range(int(images_embeddings.size()[0] / BATCH_SIZE)):
            input_tensor = images_embeddings[inner_batch_index * BATCH_SIZE : (inner_batch_index + 1) * BATCH_SIZE]
            if not PRETRAINED:
                text_tensor = text_embeddings[inner_batch_index * BATCH_SIZE : (inner_batch_index + 1) * BATCH_SIZE]
            gt_mask_tensor = gt_mask_total[inner_batch_index * BATCH_SIZE : (inner_batch_index + 1) * BATCH_SIZE]
            orig_images_tensor = orig_images_embeddings[inner_batch_index * BATCH_SIZE : (inner_batch_index + 1) * BATCH_SIZE]
            if not PRETRAINED:
                grayscale_cam = cam(input_tensor=input_tensor, text_tensor=text_tensor)
            else:
                if label_names[inner_batch_index] in imagenet_classes:
                    target_id = imagenet_classes.index(label_names[inner_batch_index])
                    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_id)
                else:
                    grayscale_cam = cam(input_tensor=input_tensor)

            grayscale_cam_mask = np.where(grayscale_cam < MASK_THRESHOLD, 0, 1)

            pred_bbox, pred_mask = MaskToBBox(grayscale_cam_mask, input_tensor.size(0))
            grayscale_cam_tensor = torch.from_numpy(pred_mask).to(GPU_ID)
            ious = iou_pytorch(outputs=grayscale_cam_tensor, labels=gt_mask_tensor)
            if first:
                grayscale_cam_total = grayscale_cam
                iou_total = ious.cpu().numpy()
                first = False
            else:
                grayscale_cam_total = np.append(grayscale_cam_total, grayscale_cam, axis=0)
                iou_total = np.append(iou_total, ious.cpu().numpy(), axis=0)

        grayscale_cam_total_mask = np.where(grayscale_cam_total < MASK_THRESHOLD, 0, 1)
        pred_bbox, pred_mask = MaskToBBox(grayscale_cam_total_mask, images_embeddings.size(0))

        loc_acc = (iou_total >= 0.5).sum()

        total_acc += loc_acc
        n += images_embeddings.size(0)

        if SAVE_RESULT:
            gt_bboxes, gt_masks = MaskToBBox(gt_mask_total.cpu().numpy(),  images_embeddings.size(0))
            for mask_num in range(len(grayscale_cam_total)):
                label = label_names[mask_num]
                Final_result.append([image_names_total[mask_num], label, iou_total[mask_num].item()])
                getHeatMap(grayscale_cam_total[mask_num], orig_images_embeddings[mask_num].permute(1, 2, 0).cpu().numpy(), os.path.join(SAVE_DIR, image_names_total[mask_num] + '_' + label + '.png'), pred_bbox[mask_num], gt_bboxes[mask_num])

    if i  % 10 == 0:
        print(f"Done {((i + 1) / len(loader) * 100)}%")

top1 = (total_acc / n) * 100

print(f"localization accuracy: {top1:.2f}")
# if SAVE_RESULT:
with open(os.path.join(SAVE_DIR, 'result.txt'), 'w') as f:
    f.write(str(Final_result))