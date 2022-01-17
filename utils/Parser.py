import os
import glob
from os.path import join as pjoin
import xml.etree.ElementTree as ET
import math
import scipy.io


class VOC2012parser():
    def __init__(self, base_path):
        self.base_path = base_path
        self.jpg_dir = pjoin(self.base_path, 'JPEGImages')
        self.class_dir = pjoin(self.base_path, 'ImageSets', 'Main')
        self.action_dir = pjoin(self.base_path, 'ImageSets', 'Action')
        self.annotation_dir = pjoin(self.base_path, 'Annotations')

    def getClassTrainValTextPaths(self):
        return glob.glob(pjoin(self.class_dir, '*_trainval.txt'))
    
    def getActionTrainValTextPaths(self):
        return glob.glob(pjoin(self.action_dir, '*_trainval.txt'))


    def xmlParsing(self, action_name, img_name):
        xml_path = pjoin(self.annotation_dir, img_name+'.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()

        list_with_all_boxes = []
        height = int(root.find("size/height").text)
        width = int(root.find("size/width").text)
        #get all bbox in that img corresponded with the class
        for boxes in root.iter('object'):
            if boxes.find("actions/{:s}".format(action_name)) is not None and int(boxes.find("actions/{:s}".format(action_name)).text) == 1:
                ymin, xmin, ymax, xmax = None, None, None, None

                ymin = math.ceil(float(boxes.find("bndbox/ymin").text))
                xmin = math.ceil(float(boxes.find("bndbox/xmin").text))
                ymax = math.ceil(float(boxes.find("bndbox/ymax").text))
                xmax = math.ceil(float(boxes.find("bndbox/xmax").text))

                list_with_single_boxes = [xmin, ymin, xmax, ymax]
                list_with_all_boxes.append(list_with_single_boxes)

        return height, width, list_with_all_boxes




    def parsing(self, path):

        # classname_trainval.txt
        class_name = path.split('/')[-1].split('_')[0]
        with open(path, 'r') as file:
            lines = file.readlines()

        # 1 denotes the positive samples while -1 denotes the negative samples
        img_paths = [ pjoin(self.jpg_dir, line.strip().split(' ')[0] + '.jpg') for line in lines if line.strip().split(' ')[1] != '-1']

        return class_name, img_paths

    def actionParsing(self, path):
        action_name = ' '.join(path.split('/')[-1].split('_')[:-1])
        with open(path, 'r') as file:
            lines = file.readlines()  #imageFileName objectIndex GroundTruth 
        
        img_paths = [pjoin(self.jpg_dir, line.strip().split(' ')[0] + '.jpg') for line in lines if len(line.strip().split(' ')) > 4 and line.strip().split(' ')[4] == '1']

        return action_name, img_paths

    def getClasswithImageNames(self):
        class_with_image_names = dict()

        trainval_text_paths = self.getClassTrainValTextPaths()
        for path in trainval_text_paths:
            class_name, img_paths = self.parsing(path)
            class_with_image_names[class_name] = img_paths
        return class_with_image_names
    
    def getActionwithImageNames(self):
        action_with_image_names = dict()
        
        trainval_text_paths = self.getActionTrainValTextPaths()
        for path in trainval_text_paths:
            action_name, img_paths = self.actionParsing(path)
            action_with_image_names[action_name] = img_paths
        return action_with_image_names
        
class HICOparser():
    def __init__(self, base_path):
        self.base_path = base_path
        #self.anno = scipy.io.loadmat(pjoin(self.base_path, 'anno.mat'))
        self.anno_bbox = scipy.io.loadmat(pjoin(self.base_path, 'anno_bbox.mat'))
        self.index2hoi = self.getIndex2HOI()
        self.train_image_path = pjoin(self.base_path, 'images', 'train2015')
        self.test_image_path = pjoin(self.base_path, 'images', 'test2015')


    def getIndex2HOI(self):
        n_pairs = len(self.anno_bbox['list_action']['vname_ing'])
        index2hoi = {}
        for i in range(n_pairs):
            index2hoi[i+1] = (self.anno_bbox['list_action']['nname'][i][0][0],
                              self.anno_bbox['list_action']['vname_ing'][i][0][0].replace('_', ' '))

        return index2hoi

    def getTrainFilenames(self):
        return [ i[0] for i in self.anno_bbox['bbox_train']['filename'][0]]

    def getTestFilenames(self):
        return [ i[0] for i in self.anno_bbox['bbox_test']['filename'][0]]

    def getTrainImageSizes(self):
        # (H, W, C)
        return [ (size[0]['height'][0][0][0], size[0]['width'][0][0][0], size[0]['depth'][0][0][0]) for size in self.anno_bbox['bbox_train']['size'][0]]

    def getTestImageSizes(self):
        # (H, W, C)
        return [ (size[0]['height'][0][0][0], size[0]['width'][0][0][0], size[0]['depth'][0][0][0]) for size in self.anno_bbox['bbox_test']['size'][0]]

    def getTrainGroundTruthActionIds(self):
        ground_truth_action_ids=[]
        for hois in self.anno_bbox['bbox_train']['hoi'][0]:
            actions_id = []
            #There might have several HOI in this image
            for gt_id in hois['id'][0]:
                actions_id.append(gt_id.item())
            ground_truth_action_ids.append(actions_id)

        return ground_truth_action_ids
    
    def getTestGroundTruthActionIds(self):
        ground_truth_action_ids=[]
        for hois in self.anno_bbox['bbox_test']['hoi'][0]:
            actions_id = []
            #There might have several HOI in this image
            for gt_id in hois['id'][0]:
                actions_id.append(gt_id.item())
            ground_truth_action_ids.append(actions_id)

        return ground_truth_action_ids

    def getTrainBboxes(self):
        # Return list of bboxhuman: xmin, ymin, xmax, ymax and bbox object: xmin, ymin, xmax, ymax
        bboxes_human=[]
        bboxes_object=[]
        for hois in self.anno_bbox['bbox_train']['hoi'][0]:
            if hois['invis'][0][0].item() != 1:
                bboxes = []

                #There might have several HOI in this image
                #Allocate all human bounding box
                for bbox in hois['bboxhuman'][0]:
                    if len(bbox) != 0:
                        bboxes.append((bbox['x1'][0][0].item(), bbox['y1'][0][0].item(), bbox['x2'][0][0].item(), bbox['y2'][0][0].item()))
                    else:
                        bboxes.append(())
                bboxes_human.append(bboxes)

                bboxes = []

                for bbox in hois['bboxobject'][0]:
                    if len(bbox) != 0:
                        bboxes.append((bbox['x1'][0][0].item(), bbox['y1'][0][0].item(), bbox['x2'][0][0].item(), bbox['y2'][0][0].item()))
                    else:
                        bboxes.append(())
                bboxes_object.append(bboxes)
            else:
                bboxes_human.append([])
                bboxes_object.append([])            

        return bboxes_human, bboxes_object

    def getTestBboxes(self):
        # Return list of bboxhuman: xmin, ymin, xmax, ymax and bbox object: xmin, ymin, xmax, ymax
        bboxes_human=[]
        bboxes_object=[]
        for hois in self.anno_bbox['bbox_test']['hoi'][0]:
            if hois['invis'][0][0].item() != 1:
                bboxes = []

                #There might have several HOI in this image
                #Allocate all human bounding box
                for bbox in hois['bboxhuman'][0]:
                    if len(bbox) != 0:
                        bboxes.append((bbox['x1'][0][0].item(), bbox['y1'][0][0].item(), bbox['x2'][0][0].item(), bbox['y2'][0][0].item()))
                    else:
                        bboxes.append(())
                bboxes_human.append(bboxes)

                bboxes = []

                for bbox in hois['bboxobject'][0]:
                    if len(bbox) != 0:
                        bboxes.append((bbox['x1'][0][0].item(), bbox['y1'][0][0].item(), bbox['x2'][0][0].item(), bbox['y2'][0][0].item()))
                    else:
                        bboxes.append(())
                bboxes_object.append(bboxes)
            else:
                bboxes_human.append([])
                bboxes_object.append([])            

        return bboxes_human, bboxes_object

    def getTrainFeatswithImageNames(self):
        feat_with_image_names = dict()
        train_filenames = self.getTrainFilenames()
        bboxes_human, bboxes_object = self.getTrainBboxes()
        image_sizes = self.getTrainImageSizes()
        gt_ids = self.getTrainGroundTruthActionIds()
        for i in range(len(train_filenames)):
            feat_with_image_names[pjoin(self.train_image_path, train_filenames[i])] = {'bbox_human': bboxes_human[i],
                                                                                        'bbox_object': bboxes_object[i],
                                                                                        'sizes': image_sizes[i],
                                                                                        'ground_truth_ids': gt_ids[i]}
        return feat_with_image_names

    def getTestFeatswithImageNames(self):
        feat_with_image_names = dict()
        test_filenames = self.getTestFilenames()
        bboxes_human, bboxes_object = self.getTestBboxes()
        image_sizes = self.getTestImageSizes()
        gt_ids = self.getTestGroundTruthActionIds()
        for i in range(len(test_filenames)):
            feat_with_image_names[pjoin(self.test_image_path, test_filenames[i])] = {'bbox_human': bboxes_human[i],
                                                                                        'bbox_object': bboxes_object[i],
                                                                                        'sizes': image_sizes[i],
                                                                                        'ground_truth_ids': gt_ids[i]}
        return feat_with_image_names
        





if __name__ == '__main__':
    base_path = "/scratch3/users/seanhsia/HOI/hico_20160224_det"
    parser = HICOparser(base_path)
    hois = parser.getIndex2HOI()
    print(list(parser.getTestFeatswithImageNames().values())[:10])
    #print(len(parser.getBboxes()[0]) == len(parser.getTrainFilenames()))
    #print(parser.getBboxes()[0][:10])
    #print(parser.getBboxes()[1][:10])
