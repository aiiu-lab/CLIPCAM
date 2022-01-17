import pandas as pd
import numpy as np
from random import sample
import os
from PIL import Image

openimage_seen_classes = ['jeans', 'suit', 'sunglasses', 'desk', 'paddle', 'microphone', 'sandal', 'mobile phone', 'ice cream', 'rifle', 'pizza', 'canoe', 'lemon', 'bookcase', 'drum', 'bee', 'parachute', 'duck', 'strawberry', 'vase', 'mushroom', 'computer keyboard', 'orange', 'goose', 'plate', 'lion', 'necklace', 'candle', 'cucumber', 'football helmet', 'lipstick', 'pig', 'balloon', 'tank', 'sink', 'couch', 'pillow', 'leopard', 'sea lion', 'barrel', 'cowboy hat', 'zucchini', 'television', 'jellyfish', 'filing cabinet', 'ski', 'otter', 'teapot', 'bagel', 'bell pepper', 'snail', 'limousine',
                          'tripod', 'saxophone', 'miniskirt', 'castle', 'cabbage', 'wok', 'cheetah', 'hamster', 'violin', 'polar bear', 'washing machine', 'cello', 'broccoli', 'umbrella', 'zebra', 'ostrich', 'scarf', 'microwave oven', 'tent', 'refrigerator', 'porcupine', 'accordion', 'sewing machine', 'backpack', 'infant bed', 'lynx', 'racket', 'ladybug', 'traffic light', 'magpie', 'printer', 'fountain', 'starfish', 'pineapple', 'guacamole', 'gondola', 'goldfish', 'wall clock', 'trombone', 'coffeemaker', 'tiger', 'computer mouse', 'missile', 'cannon', 'artichoke', 'banana', 'burrito', 'teddy bear']
openimage_unseen_classes = ['person', 'clothing', 'mammal', 'human body', 'car', 'wheel', 'human hair', 'plant', 'man', 'human head', 'tree', 'human arm', 'human face', 'woman', 'human nose', 'flower', 'tire', 'girl', 'human hand', 'human mouth', 'human eye', 'footwear', 'building', 'human leg', 'land vehicle', 'food', 'dog', 'vehicle', 'sports equipment', 'fashion accessory', 'snack', 'window', 'baked goods', 'airplane', 'vehicle registration plate', 'dress', 'human ear', 'dessert', 'furniture', 'table', 'bird', 'boy', 'house', 'boat', 'auto part', 'fruit', 'fast food', 'vegetable',
                            'tableware', 'fish', 'drink', 'dairy product', 'cat', 'helmet', 'chair', 'toy', 'truck', 'houseplant', 'horse', 'cake', 'salad', 'bread', 'glasses', 'bicycle', 'bicycle wheel', 'sports uniform', 'carnivore', 'seafood', 'trousers', 'animal', 'bottle', 'ball', 'door', 'poster', 'human foot', 'insect', 'shelf', 'sculpture', 'flowerpot', 'pastry', 'van', 'human beard', 'musical instrument', 'swimwear', 'shorts', 'swimming pool', 'monkey', 'jacket', 'aircraft', 'coffee cup', 'weapon', 'cocktail', 'pasta', 'motorcycle', 'juice', 'hat', 'cabinetry', 'tower', 'camera', 'helicopter']


def process_seen_unseen(search_classname_func, detection_data):
    selected_data_list = []

    detection_data['ClassName'] = detection_data['LabelName'].apply(
        search_classname_func)

    seen_detection_data = detection_data.loc[detection_data['ClassName'].isin(
        openimage_seen_classes)]
    unseen_detection_data = detection_data.loc[detection_data['ClassName'].isin(
        openimage_unseen_classes)]
    seen_detection_data['Type'] = 'seen'
    unseen_detection_data['Type'] = 'unseen'

    selected_data_list += process_classes(
        openimage_seen_classes, seen_detection_data, 224, 224)
    selected_data_list += process_classes(
        openimage_unseen_classes, unseen_detection_data, 224, 224)

    selected_data = pd.concat(selected_data_list, ignore_index=True)
    selected_data = selected_data.reindex(
        np.random.permutation(selected_data.index))

    return selected_data


def process_classes(classes, data, h, w):
    l = []
    for seen_class in classes:
        curr_data = data.loc[data['ClassName'] == seen_class]
        images = sample(curr_data['ImageID'].unique().tolist(), 10)
        for image_id in images:
            image_data = curr_data.loc[curr_data['ImageID'] == image_id]
            mask_total = np.zeros((w, h))
            for index, mask in image_data.iterrows():
                xmin, xmax, ymin, ymax = int(
                    mask['XMin'] * w),  int(mask['XMax'] * w),  int(mask['YMin'] * h),  int(mask['YMax'] * h)
                mask_total[ymin: ymax+1, xmin:xmax+1] = 1
            filtered_gt_mask_cordinates = np.array(np.where(mask_total == 1))
            gt_total_mask_bbox = np.array((np.min(filtered_gt_mask_cordinates[1]), np.min(
                filtered_gt_mask_cordinates[0]), np.max(filtered_gt_mask_cordinates[1]), np.max(filtered_gt_mask_cordinates[0])))
            xmin, ymin, xmax, ymax = gt_total_mask_bbox[0], gt_total_mask_bbox[
                1], gt_total_mask_bbox[2], gt_total_mask_bbox[3]
            append_data = image_data.head(1)
            append_data['XMin'], append_data['YMin'], append_data['XMax'], append_data['YMax'] = xmin, ymin, xmax, ymax
            l.append(append_data)
    return l


def getImage(data_dir, image_id):
    image_path = os.path.join(data_dir, image_id + '.jpg')
    image = Image.open(image_path)
    image = image.resize((224, 224))
    return image


def getGTMask(row, w, h):
    xmin, xmax, ymin, ymax = int(row['XMin']),  int(
        row['XMax']),  int(row['YMin']),  int(row['YMax'])
    gt_total_mask = np.zeros((w, h)).astype('uint8')
    gt_total_mask[ymin: ymax+1, xmin:xmax+1] = 1
    return gt_total_mask

def bboxs2Mask(bbx, h, w):
    mask = np.zeros((224, 224)).astype('uint8')
    xmin, ymin, xmax, ymax = bbx
    xmin, xmax, ymin, ymax = int(xmin * 224.0 / w), int(xmax * 224.0 / w), int(ymin * 224.0 / h),  int(ymax * 224.0 / h)
    mask[ymin: ymax+1, xmin:xmax+1] = 1
    return mask
