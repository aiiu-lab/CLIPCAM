import clip_modified
from tqdm.notebook import tqdm
import os
import numpy as np
from torchvision import transforms
import argparse
import pickle

from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.openimage_utils import *
from utils.imagenet_utils import *
from utils.grid_utils import *
from utils.evaluation_tools import *
from utils.preprocess import getAttacker
from utils.dataset import OpenImageDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='datasets/OpenImage/validation',
                    help="directory of openimage dataset")
parser.add_argument("--save_dir", type=str, default='eval_result/rn50-grad',
                    help="directory to save the result")
parser.add_argument("--gpu_id", type=int, default=1,
                    help="GPU to run on")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--mask_threshold", type=float, default=0.2,
                    help="Threshold of the localization mask")
parser.add_argument("--sentence_prefix", type=str, default='',
                    help="Text input prefix: \"PREFIX\" + \"object class name\"")
parser.add_argument("--attack", type=str, default='None',
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--save_result", type=int, default=0,
                    help="save result or not, type 1 or 0")
args = parser.parse_args()

DATA_DIR = args.data_dir
SAVE_DIR = args.save_dir
SENTENCE_PREFIX = args.sentence_prefix
GPU_ID = args.gpu_id
CLIP_MODEL_NAME = args.clip_model_name
CAM_MODEL_NAME = args.cam_model_name
MASK_THRESHOLD = args.mask_threshold
ATTACK = args.attack
SAVE_RESULT = args.save_result
RESIZE = True

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

dataset = OpenImageDataset(data_dir=DATA_DIR, transform=ImageTransform,
                           original_transform=originalTransform, gpu_id=GPU_ID, attack=ATTACK)
detection_data = dataset.detection_data
class_names = dataset.class_names

selected_data = process_seen_unseen(dataset.searchClassName, detection_data)

total_count = 0

total_result = []
for chnk_data in chunker(selected_data, 4, 0):
    chnk_data = chnk_data.reset_index()
    grid = []
    for index, row in chnk_data.iterrows():
        img = getImage(os.path.join(DATA_DIR, 'data'), row['ImageID'])
        mask = getGTMask(row, 224, 224)
        # transforms.ToPILImage()(ori_img).convert('RGB').save(os.path.join(SAVE_DIR, row['ImageID'] + str(index) + '.png'))
        mask_im = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
        gt_seen_unseen = row['Type']
        classname = row['ClassName'].lower()
        grid.append([img, classname, gt_seen_unseen, mask, mask_im])

    img_grid = get_concat(grid[0][0], grid[1][0], grid[2][0], grid[3][0])

    # img_grid.save(os.path.join(SAVE_DIR, 'img_grid.png'))

    img_grid_small = img_grid.resize((224, 224))

    if ATTACK != '':
        img_grid_small = getAttacker(
            img_grid_small, type=ATTACK, gpu_id=GPU_ID)

    img_grid_small_tensor = ImageTransform(
        img_grid_small).unsqueeze(0).to(GPU_ID)

    for i in range(4):
        sentence = [f'{SENTENCE_PREFIX}{grid[i][1]}']
        mask_im = get_cat_gt_masks(grid, i)
        # mask_im.save(os.path.join(SAVE_DIR, str(total_count) + '_mask_im.png'))
        mask_np = (np.array(mask_im) / 255).astype(np.uint8)
        _, mask_bboxes = get_4_bbox(mask_np)
        if not PRETRAINED:
            text_token = clip_modified.tokenize(
                sentence).to(GPU_ID)  # tokenize
            text_embedding = model.encode_text(
                text_token)  # embed with text encoder
        if not PRETRAINED:
            grayscale_cam = cam(
                input_tensor=img_grid_small_tensor, text_tensor=text_embedding)
        else:
            if grid[i][2] == 'seen':
                target_id = imagenet_classes.index(grid[i][1])
                grayscale_cam = cam(
                    input_tensor=img_grid_small_tensor, target_category=target_id)
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

        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam_mask = np.where(grayscale_cam < MASK_THRESHOLD, 0, 1)
        circular_mask = create_circular_mask(h=224, w=224, radius=30)
        grayscale_cam_mask = np.where(circular_mask > 0, 0, grayscale_cam_mask)
        grayscale_cam_img = Image.fromarray(
            (grayscale_cam_mask * 255).astype(np.uint8)).convert('L')

        grayscale_cam_img = grayscale_cam_img.resize((448, 448))
        # grayscale_cam_img.save(os.path.join(SAVE_DIR, str(total_count) + '_graycam.png'))
        grayscale_cam_np = (np.array(grayscale_cam_img) / 255).astype(np.uint8)
        total_pred_mask, total_bboxes = get_4_bbox(grayscale_cam_np)
        # total_pred_mask.save(os.path.join(SAVE_DIR, str(total_count) + '_total_pred_mask.png'))

        total_pred_mask_np = (np.array(total_pred_mask) / 255).astype(np.uint8)
        iou = iou_numpy(total_pred_mask_np[:, :, 0], mask_np[:, :, 0])
        if SAVE_RESULT:
            getHeatMap4bboxes(grayscale_cam, img_grid_small, os.path.join(SAVE_DIR, str(
                total_count) + '_' + grid[i][1] + '.png'), total_bboxes, mask_bboxes)
        # print(grid[i][1], grid[i][2], "{:.4f}".format(iou))
        total_result.append(
            [grid[i][1], grid[i][2], float("{:.4f}".format(iou))])
        total_count += 1
    # if total_count % 10 == 0:
    #     print(grid[i][1], grid[i][2], "{:.4f}".format(iou))
    # if total_count >= 8:
    #     break
print(f"mIoU = {sum([i[2] for i in total_result]) / len(total_result)}")

with open(os.path.join(SAVE_DIR, 'result.txt'), 'w') as f:
    f.write(str(total_result))


def filter_imagenet(l, mode):
    new_l = []
    for item in l:
        if mode == 'seen':
            if item[1] == 'seen':
                new_l.append(item)
        elif mode == 'unseen':
            if item[1] == 'unseen':
                new_l.append(item)
    return new_l


result_seen = filter_imagenet(total_result, 'seen')
result_unseen = filter_imagenet(total_result, 'unseen')
print(f"seen mIoU = {sum([i[2] for i in result_seen]) / len(result_seen)}")
print(
    f"unseen mIoU = {sum([i[2] for i in result_unseen]) / len(result_unseen)}")
