# CLIPCAM: Zero-shot Text-guided Object and Action Localization
Official implementation of paper CLIPCAM: 

## Installation
1. install pytorch 1.9.0, torchvision 0.10.0 with compatible cuda version  
`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
2. install required package  
`pip install -r requirements.txt`

## Usage
### OpenImage grid-view seen/unseen object localization  
1. Dataset structure  
```
|--OpenImage
    |--validation
        |--data
            |--{image_path_1}
            |--{image_path_2}
            |-- ...
        |--labels
            |--detections.csv
        |--metadata
            |--classes.csv
```
2. Run `evaluate_grid_openimage.py`  
```
python evaluate_grid_openimage.py \
    --data_dir .\Dataset\OpenImage\validation \
    --gpu_id 0 \
    --clip_model_name 'ViT-B/32' \
    --cam_model_name 'GradCAM' \
    --save_dir 'eval_result/vitb32' \
    --mask_threshold 0.2 \
    --sentence_prefix 'a photo of ' \
    --attack 'None' \
    --save_result 1
```

### HICO-DET grid-view action localization  
1. Dataset structure  
```
|--HICO-DET
    |--images
        |--test
            |--{image_path_1}
            |--{image_path_2}
            |-- ...
        |--train
            |--{image_path_1}
            |--{image_path_2}
            |-- ...
    |--anno.mat
    |--anno_bbox.mat
```
2. Run `verb_grid.py` for pre-trained model
```
python verb_grid.py \
    --data_dir .\Dataset\OpenImage\validation \
    --gpu_id 0 \
    --clip_model_name 'ViT-B/32-pretrained' \
    --cam_model_name 'GradCAM_original' \
    --save_dir 'eval_result/vitb32-pretrained' \
    --mask_threshold 0.2 \
    --train_mode 'half' \
    --model_name 'model.pth' \
    --save_result 1
```
3. Run `verb_grid.py` for CLIPCAM
```
python verb_grid.py \
    --data_dir .\Dataset\HICO-DET \
    --gpu_id 0 \
    --clip_model_name 'ViT-B/32' \
    --cam_model_name 'GradCAM' \
    --save_dir 'eval_result/vitb32-pretrained' \
    --mask_threshold 0.2 \
    --sentence_prefix 'someone is ' \
    --train_mode 'half' \
    --save_result 1
```

### ImageNet evaluation on validation set  
1. Dataset structure  
```
|--ImageNet
    |--validation
        |--{label_1}\
            |--{image_path_1}
            |--{image_path_2}
            |-- ...
        |--{label_2}
        |-- ...
    |--bbox
        |--val
            |--{image_path_1}.xml
            |--{image_path_1}.xml
            |-- ...
```
2. Run `evaluate_imagenet.py`  
```
python evaluate_imagenet.py \
    --data_dir .\Dataset\ImageNet\validation \
    --gpu_id 0 \
    --clip_model_name 'ViT-B/32' \
    --cam_model_name 'GradCAM' \
    --save_dir 'eval_result/vitb32' \
    --batch 128 \
    --save_result 1 \
    --sentence_prefix 'sentence' \
    --attack 'None'
```
### OpenImage seen/unseen object localization evaluation
1. Run `evaluate_openimage.py`  
```
python evaluate_openimage.py \
    --data_dir {path to dataset} \
    --gpu_id 0 \
    --clip_model_name 'ViT-B/32' \
    --cam_model_name 'GradCAM' \
    --save_dir 'eval_result/vitb32' 
    --save_result 1 \
    --sentence_prefix 'a photo of ' \
    --distill_num 2 \
    --attack 'None'
```

### OpenImage, COSMOS and other evaluation
1. Run `evaluate.py`  
```
python evaluate.py \
    --data_dir {path to dataset} \
    --gpu_id 0 \
    --clip_model_name 'ViT-B/32' \
    --save_dir 'eval_result/vitb32' 
    --save_result 1 \
    --sentence_prefix 'a photo of ' \
    --distill_num 2
```

## Citing
If you find the paper or the code useful for your study, please consider citing the CLIPCAM paper:
```bash
@article{,
   author = {},
    title = "{}",
    journal = {},
    year = 
}
```