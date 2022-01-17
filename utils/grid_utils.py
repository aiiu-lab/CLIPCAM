from PIL import Image, ImageDraw
import numpy as np
from .evaluation_tools import MaskToBBox
import cv2


def chunker(seq, size, overlap):
    for pos in range(0, len(seq), size-overlap):
        yield seq.iloc[pos:pos + size]


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_concat(im1, im2, im3, im4):
    dst1 = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst1.paste(im1, (0, 0))
    dst1.paste(im2, (im1.width, 0))
    dst2 = Image.new('RGB', (im3.width + im4.width, im3.height))
    dst2.paste(im3, (0, 0))
    dst2.paste(im4, (im3.width, 0))
    dst = Image.new('RGB', (dst1.width, dst1.height + dst2.height))
    dst.paste(dst1, (0, 0))
    dst.paste(dst2, (0, dst1.height))
    return dst


def get_cat_gt_masks(grid, turn, data_dir):
    negative_mask = Image.fromarray(
        np.zeros((224, 224)).astype('uint8')).convert('L')
    l = [negative_mask, negative_mask, negative_mask, negative_mask]
    for i in range(4):
        if grid[i][1] == grid[turn][1]:
            l[i] = grid[i][4]
    mask_im = get_concat(l[0], l[1], l[2], l[3])
    return mask_im


def get_4_bbox(im):
    map1 = im[0:224, 0:224]
    map2 = im[0:224, 224:448]
    map3 = im[224:448, 0:224]
    map4 = im[224:448, 224:448]
    bboxes, pred_masks = MaskToBBox(np.array([map1, map2, map3, map4]), 4)
    pred_mask_1_img = Image.fromarray(
        (pred_masks[0] * 255).astype(np.uint8)).convert('L')
    pred_mask_2_img = Image.fromarray(
        (pred_masks[1] * 255).astype(np.uint8)).convert('L')
    pred_mask_3_img = Image.fromarray(
        (pred_masks[2] * 255).astype(np.uint8)).convert('L')
    pred_mask_4_img = Image.fromarray(
        (pred_masks[3] * 255).astype(np.uint8)).convert('L')
    bboxes[1][0] += 224
    bboxes[1][2] += 224
    bboxes[2][1] += 224
    bboxes[2][3] += 224
    bboxes[3][0] += 224
    bboxes[3][1] += 224
    bboxes[3][2] += 224
    bboxes[3][3] += 224
    bboxes = (bboxes / 2).astype(np.uint8)
    total_pred_mask = get_concat(
        pred_mask_1_img, pred_mask_2_img, pred_mask_3_img, pred_mask_4_img)
    return total_pred_mask, bboxes


def getHeatMap4bboxes(mask, img, filename, bboxes, gt_bbox):
    img.save(filename.split('.')[0] + '_ori.png')
    img = np.array(img) / 255
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

    for i in range(4):
        draw.rectangle((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]), fill=(
            0, 0, 0, 0), outline=(255, 0, 0), width=4)
        draw.rectangle((gt_bbox[i][0], gt_bbox[i][1], gt_bbox[i][2], gt_bbox[i][3]), fill=(
            0, 0, 0, 0), outline=(0, 0, 255), width=4)
    heatmap_im.save(filename)


def getPredictionRegion(l):
    if max(l) == l[0]:
        return 0
    elif max(l) == l[1]:
        return 1
    elif max(l) == l[2]:
        return 2
    elif max(l) == l[3]:
        return 3