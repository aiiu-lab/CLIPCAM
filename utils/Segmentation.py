import glob
from os.path import join as pjoin

import numpy as np
from PIL import Image
import cv2



class BaseSegmentation:
    def computeIOU(self, pred, mask, threshold=0.5):
        smooth = 1e-6
        pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
        pred_mask = np.where(pred>threshold, 1, 0)
        intersection = (pred_mask & mask).sum((0, 1))
        union = (pred_mask | mask).sum((0, 1))
        iou = (intersection + smooth) / (union + smooth)

        return iou
    
    def bboxs2Mask(self, bboxs, img_height, img_width):
        mask = np.zeros((img_height, img_width)).astype('uint8')
        for xmin, ymin, xmax, ymax in bboxs:
            mask[ymin: ymax+1, xmin:xmax+1] = 1
        return mask


class VOCSegmentationPreprocess(BaseSegmentation):
    def __init__(self, base_path):
        self.base_path = base_path
        self.segment_dir = pjoin(self.base_path, 'SegmentationClass')
        self.annotation_dir = pjoin(self.base_path, 'Annotations')

        self.class2code={
            'background': 0,
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20
        }
        self.code2class = [ k for k in self.class2code.keys()]
        self.segmentation_filenames = [ path.split('/')[-1].split('.')[0] for path in glob.glob(pjoin(self.segment_dir, '*.png'))]
        self.annotation_filenames = [ path.split('/')[-1].split('.')[0] for path in glob.glob(pjoin(self.annotation_dir, '*.xml'))]

    
    def getMaskOfClassname(self, classname, img_name):
        img_path = pjoin(self.segment_dir, img_name+'.png')
        img = Image.open(img_path)
        class_colorcode = self.class2code[classname]

        img = np.array(img)
        mask = np.where(img != class_colorcode, 0, 1)
        return mask
    





if __name__ == '__main__':
    base_path = "/scratch3/users/seanhsia/VOC2012/VOCdevkit/VOC2012"
    segmentation_preprocess = VOCSegmentationPreprocess(base_path)

    mask = segmentation_preprocess.getMaskOfClassname('person' ,'2007_000032')
    
    print(mask.shape)
    img = Image.fromarray(np.uint8(mask * 255))
    img = img.convert('L')
    img = img.save('test.png')
    