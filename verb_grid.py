import clip_modified
#from tqdm.notebook import tqdm
from tqdm import tqdm
import os
#import pandas as pd
import pickle
import numpy as np
from random import sample
from torchvision import transforms
import argparse
import random

from utils.model import getCLIP, getCAM, getFineTune
from utils.preprocess import getImageTranform
from utils.openimage_utils import *
from utils.imagenet_utils import *
from utils.grid_utils import *
from utils.evaluation_tools import *
from utils.dataset import HICODataset, HICO_filtered_actions


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='/scratch2/users/jason/Dataset/hico_20160224_det',
                    help="directory of hico")
parser.add_argument("--save_dir", type=str, default='verb_result',
                    help="directory to save the result")
parser.add_argument("--gpu_id", type=int, default=1,
                    help="GPU to work on")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50-pretrained', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM_original', help="Model name of GradCAM")
parser.add_argument("--mask_threshold", type=float, default=0.2,
                    help="Threshold of the mask")
# parser.add_argument("--sentence_prefix", type=str, default='',
#                     help="select input of the prefix")
parser.add_argument("--train_mode", type=str, default='half',
                    help="modes to load dataset: full, half, few")
parser.add_argument("--model_name", type=str, default='model',
                    help="pretrained model name in checkpoint/models")
parser.add_argument("--save_result", type=int, default=0,
                    help="save result or not")
args = parser.parse_args()

DATA_DIR = args.data_dir
SAVE_DIR = args.save_dir
# SENTENCE_PREFIX = args.sentence_prefix
GPU_ID = args.gpu_id
CLIP_MODEL_NAME = args.clip_model_name
CAM_MODEL_NAME = args.cam_model_name
MASK_THRESHOLD = args.mask_threshold
SAVE_RESULT = args.save_result
TRAIN_MODE = args.train_mode
MODEL_NAME = args.model_name
RESIZE = True
os.makedirs(SAVE_DIR, exist_ok=True)

if CLIP_MODEL_NAME.split('-')[-1] == 'pretrained':
    PRETRAINED = True
else:
    PRETRAINED = False


model, target_layer, reshape_transform = getCLIP(
    model_name=CLIP_MODEL_NAME, gpu_id=GPU_ID)

cam = getCAM(model_name=CAM_MODEL_NAME, model=model, target_layer=target_layer,
             gpu_id=GPU_ID, reshape_transform=reshape_transform)

ImageTransform = getImageTranform(resize=RESIZE)
originalTransform = getImageTranform(resize=RESIZE, normalized=False)

test_dataset = HICODataset(
    DATA_DIR, ImageTransform, originalTransform, split='test', mode=TRAIN_MODE)
if PRETRAINED:
    dataset = HICODataset(DATA_DIR, ImageTransform,
                            originalTransform, split='train', mode=TRAIN_MODE)
    total_actions = sorted(
        list(set(dataset.gt_actions + test_dataset.gt_actions)))
        # total_actions = dataset.gt_actions
    model = getFineTune(model_name=CLIP_MODEL_NAME,
                        model=model, out_feature=len(total_actions))
    model.load_state_dict(torch.load(
        os.path.join('checkpoints', 'models', MODEL_NAME)))
    model = model.to(GPU_ID)

raw_data = test_dataset.data_list

random.shuffle(raw_data)
data = []
flag = True
while(len(data) < 500 and flag):
    flag = False
    grid = []
    grid_ori = []
    pool1, pool2 = [], []
    for rd in raw_data:
        rd_ori = rd
        mask_object = bboxs2Mask(rd[4][0], rd[1], rd[2])
        mask_human = bboxs2Mask(rd[4][0], rd[1], rd[2])
        mask = (mask_human | mask_object)
        mask_im = Image.fromarray(
            (mask * 255).astype(np.uint8)).convert('L')
        rd = [getImage(DATA_DIR, rd[0].split('.')[0]), test_dataset.index2hoi[rd[5]][1], test_dataset.index2hoi[rd[5]][0], mask, mask_im]
        if not grid:
            grid.append(rd)
            grid_ori.append(rd_ori)
            pool1.append(rd[1])
            pool2.append(rd[2])
        else:
            if rd[1] not in pool1 and rd[2] not in pool2:
                grid.append(rd)
                grid_ori.append(rd_ori)
                pool1.append(rd[1])
                pool2.append(rd[2])
                if len(grid) == 4:
                    flag = True
                    break
    if flag:
        data.append(grid)
        for tmp in grid_ori:
            raw_data.remove(tmp)


total_count = 0
region_correct = 0
seen = 0
unseen = 0
seen_count = 0
unseen_count = 0
iou_seen = []
iou_unseen = []
total_result = []
pbar = tqdm(data)
for grid in pbar:  # grid = [img, verb, noun, mask, mask_im]
    pbar.refresh()
    
    img_grid = get_concat(grid[0][0], grid[1][0], grid[2][0], grid[3][0])

    #img_grid.save(os.path.join(SAVE_DIR, 'img_grid.png'))

    img_grid_small = img_grid.resize((224, 224))
    img_grid_small_tensor = ImageTransform(
        img_grid_small).unsqueeze(0).to(GPU_ID)
    for i in range(4):
        #sentence = [f'{SENTENCE_PREFIX}{grid[i][2]}']
        # sentence = [f'{grid[i][2]} {grid[i][1]}']
        if PRETRAINED:
            try:
                ground_trouth_action_id = total_actions.index(grid[i][2])
            except:
                ground_trouth_action_id = -1
        else:
            sentence = [f'Someone is {grid[i][2]}']
        mask_im = get_cat_gt_masks(grid, i, DATA_DIR)
        # mask_im.save(os.path.join(SAVE_DIR, str(total_count) + '_mask_im.png'))
        mask_np = (np.array(mask_im) / 255).astype(np.uint8)
        _, mask_bboxes = get_4_bbox(mask_np)

        # text_token = clip_modified.tokenize(sentence).to(GPU_ID) #tokenize
        # text_embedding = model.encode_text(text_token) #embed with text encoder
        if PRETRAINED:
            if ground_trouth_action_id != -1:
                grayscale_cam = cam(
                    input_tensor=img_grid_small_tensor, target_category=ground_trouth_action_id)
            else:
                output = model(img_grid_small_tensor)
                logits = torch.nn.functional.softmax(output, dim=-1)
                pred_categories = logits.topk(4, 1, True, True)[1].t()
                stacked_image = [img_grid_small_tensor for i in range(4)]
                stacked_image = torch.stack(stacked_image, dim=1)[0].to(GPU_ID)
                stacked_mask = np.array([mask_np[:, :, 0] for i in range(4)])
                grayscale_cam = cam(input_tensor=stacked_image,
                                    target_category=pred_categories.tolist())
                grayscale_cam_mask = np.where(
                    grayscale_cam < MASK_THRESHOLD, 0, 1)
                top = -np.Inf
                top_index = 0
                for pred_order in range(4):
                    m = grayscale_cam_mask[pred_order]
                    top_left, top_right, bot_left, bot_right = np.mean(m[:224, :224]), np.mean(
                        m[:224, 224:]), np.mean(m[224:, :224]), np.mean(m[224:, 224:])
                    if i == 0:
                        t = top_left - (top_right + bot_left + bot_right)
                    elif i == 1:
                        t = top_right - (top_left + bot_left + bot_right)
                    elif i == 2:
                        t = bot_left - (top_right + top_left + bot_right)
                    elif i == 3:
                        t = bot_right - (top_right + bot_left + top_left)
                    if t > top:
                        top_index = pred_order
                        top = t
                grayscale_cam = np.array([grayscale_cam[top_index]])
        else:
            text_token = clip_modified.tokenize(
                sentence).to(GPU_ID)  # tokenize
            text_embedding = model.encode_text(
                text_token)  # embed with text encoder
            grayscale_cam = cam(
                input_tensor=img_grid_small_tensor, text_tensor=text_embedding)

        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam_mask = np.where(grayscale_cam < MASK_THRESHOLD, 0, 1)
        #circular_mask = create_circular_mask(h = 224, w = 224, radius=30)
        #grayscale_cam_mask = np.where(circular_mask > 0, 0, grayscale_cam_mask)
        grayscale_cam_img = Image.fromarray(
            (grayscale_cam_mask * 255).astype(np.uint8)).convert('L')

        grayscale_cam_img = grayscale_cam_img.resize((448, 448))
        # grayscale_cam_img.save(os.path.join(SAVE_DIR, str(total_count) + '_graycam.png'))
        grayscale_cam_np = (np.array(grayscale_cam_img) / 255).astype(np.uint8)
        total_pred_mask, total_bboxes = get_4_bbox(grayscale_cam_np)
        #total_pred_mask.save(os.path.join(SAVE_DIR, str(total_count) + '_total_pred_mask.png'))

        # total_pred_mask_np = (np.array(total_pred_mask) / 255).astype(np.uint8)
        #iou = iou_numpy(total_pred_mask_np[:, :, 0], mask_np[:, :, 0])
        iou = iou_numpy(grayscale_cam_np[:, :], mask_np[:, :, 0])

        top_left, top_right, bot_left, bot_right = np.mean(grayscale_cam_np[:224, :224]), np.mean(
            grayscale_cam_np[:224, 224:]), np.mean(grayscale_cam_np[224:, :224]), np.mean(grayscale_cam_np[224:, 224:])

        prediction_region = getPredictionRegion(
            [top_left, top_right, bot_left, bot_right])
        if i == prediction_region:
            region_correct += 1
            if grid[i][2] in HICO_filtered_actions:
                seen += 1
            else:
                unseen += 1

        if grid[i][2] in HICO_filtered_actions:
            seen_count += 1
            iou_seen.append(iou)
        else:
            unseen_count += 1
            iou_unseen.append(iou)

        if SAVE_RESULT:
            getHeatMap4bboxes(grayscale_cam, img_grid_small, os.path.join(
                SAVE_DIR, f"{total_count}_{grid[i][1]}_{grid[i][2]}.png"), total_bboxes, mask_bboxes)
        total_result.append(
            [grid[i][1], grid[i][2], float("{:.4f}".format(iou))])
        total_count += 1

print(f"mIoU = {sum([i[2] for i in total_result]) / len(total_result)}")
print(f"mIoU seen = {sum(iou_seen) / seen_count}")
print(f"mIoU unseen = {sum(iou_unseen) / unseen_count}")
print(f"region accuracy = {region_correct / total_count}")
print(f"seen accuracy = {seen / seen_count}")
print(f"unseen accuracy = {unseen / unseen_count}")
