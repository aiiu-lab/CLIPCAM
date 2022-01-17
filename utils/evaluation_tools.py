import torch
import clip_modified
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def zeroshot_classifier(classnames, templates, model, GPU_ID):
    with torch.no_grad():
        zeroshot_weights = []
        zeroshot_weights_text = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            class_texts = [classname]
            texts = clip_modified.tokenize(texts).to(GPU_ID)  # tokenize
            class_texts = clip_modified.tokenize(
                class_texts).to(GPU_ID)  # tokenize
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            class_text_embeddings = model.encode_text(
                class_texts)  # embed with text encoder
            class_text_embeddings /= class_text_embeddings.norm(
                dim=-1, keepdim=True)
            class_text_embeddings = class_text_embeddings.mean(dim=0)
            class_text_embeddings /= class_text_embeddings.norm()
            zeroshot_weights_text.append(class_text_embeddings)
        class_sentences = torch.stack(zeroshot_weights, dim=0).to(GPU_ID)
        class_words = torch.stack(zeroshot_weights_text, dim=0).to(GPU_ID)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(GPU_ID)
    return zeroshot_weights, class_sentences, class_words


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_indexes = (correct.t() == True).nonzero(as_tuple=False)
    correct_indice_topk = correct_indexes.t()[0]
    correct_indice_top1 = (correct_indexes[:, 1] == 0).nonzero(
        as_tuple=False).t()[0]
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], [correct_indice_top1, correct_indice_topk]


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))         # Will be zero if both are 0
    SMOOTH = 1e-6
    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou  # Or thresholded.mean() if you are interested in average across the batch


def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou  # Or thresholded.mean()


def getHeatMap(mask, img, filename, bbox, gt_bbox, pred_text=None, gt_text=None):
    img_im = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
    img_im.save(filename.split('.')[0] + '_ori.png')
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    heatmap = np.uint8(255 * cam)
    heatmap_im = Image.fromarray(heatmap).convert('RGB')
    draw = ImageDraw.Draw(heatmap_im, 'RGBA')

    draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]),
                   fill=(0, 0, 0, 0), outline=(255, 0, 0), width=4)
    draw.rectangle((gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]), fill=(
        0, 0, 0, 0), outline=(0, 0, 255), width=4)
    font = ImageFont.truetype("utils/FreeMono.ttf", 18)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if pred_text != None:
        draw.text((bbox[0], bbox[1]), pred_text, (255, 255, 255),
                  font=font, stroke_width=2, stroke_fill=(255, 0, 0))
    if gt_text != None:
        draw.text((gt_bbox[0], gt_bbox[1]), gt_text, (255, 255, 255),
                  font=font, stroke_width=2, stroke_fill=(0, 0, 255))
    heatmap_im.save(filename)


def getHeatMapNoBBox(mask, img, filename):
    img_im = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
    img_im.save(filename.split('.')[0] + '_ori.png')
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    heatmap = np.uint8(255 * cam)
    heatmap_im = Image.fromarray(heatmap).convert('RGB')
    heatmap_im.save(filename)


def MaskToBBox(masks, size):

    # masks_connected = []
    # for index in range(len(masks)):
    #     filtered_cam_cv2 = np.array(masks[index] * 255).astype('uint8')
    #     filtered_cam_cv2 = cv2.cvtColor(filtered_cam_cv2, cv2.COLOR_GRAY2BGR)
    #     filtered_cam_cv2 = cv2.cvtColor(filtered_cam_cv2, cv2.COLOR_BGR2GRAY)
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(filtered_cam_cv2, connectivity=4)
    #     max_index = np.argmax(stats[:, -1]) + 1
    #     filtered_cam_cv2 = np.where(output == max_index, 1, 0)
    #     masks_connected.append(filtered_cam_cv2)

    # masks = np.array(masks_connected)
    # print(masks.shape)
    filtered_cams = np.array(np.where(masks == 1))
    pred_bboxes = []
    for index in range(len(masks)):
        filtered_cam_cords = filtered_cams.transpose(
        )[np.where(filtered_cams[0] == index)].transpose()
        if len(filtered_cam_cords[1]) != 0:
            pred_bbox = np.array((np.min(filtered_cam_cords[2]), np.min(
                filtered_cam_cords[1]), np.max(filtered_cam_cords[2]), np.max(filtered_cam_cords[1])))
        else:
            pred_bbox = np.array((0, 0, 0, 0))
        pred_bboxes.append(pred_bbox)

    pred_bboxes = np.array(pred_bboxes)

    pred_mask = np.zeros((size, 224, 224)).astype('uint8')
    for bbox_index in range(len(pred_bboxes)):
        xmin, ymin, xmax, ymax = pred_bboxes[bbox_index][0], pred_bboxes[
            bbox_index][1], pred_bboxes[bbox_index][2], pred_bboxes[bbox_index][3]
        pred_mask[bbox_index][ymin: ymax+1, xmin:xmax+1] = 1
    return pred_bboxes, pred_mask
