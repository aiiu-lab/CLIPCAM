from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2
from attacks.fog import fog_creator
from attacks.snow import snow_creator, make_kernels

def getImageTranform(resize = True, normalized = True):
    if resize:
        if normalized:
            return Compose([
                Resize((224, 224), interpolation=Image.BICUBIC),
                CenterCrop((224, 224)),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            return Compose([
                Resize((224, 224), interpolation=Image.BICUBIC),
                CenterCrop((224, 224)),
                lambda image: image.convert("RGB"),
                ToTensor(),
            ])
    else: 
        if normalized:
            return Compose([
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            return Compose([
                lambda image: image.convert("RGB"),
                ToTensor(),
            ])

def getAttacker(pixel_img, type = 'None', gpu_id = 0):
    if type == 'None':
        return pixel_img
    elif type == 'fog':
        fog_vars = []
        for i in range(int(np.log2(256))):
            for j in range(3):
                var = torch.rand((1, 2**i, 2**i), device=gpu_id)
                var.requires_grad_()
                fog_vars.append(var)
        map_size = 2 ** (int(np.log2(224)) + 1)
        fog = fog_creator(fog_vars, 1, mapsize=map_size, wibbledecay=2.0, gpu_id=gpu_id)[:,:,16:-16,16:-16]
    elif type == 'snow':
        kernels = make_kernels(gpu_id=gpu_id)
        flake_intensities = torch.exp(-1./(0.02)*torch.rand(1,7,224//4,224//4)).to(gpu_id)
        flake_intensities.requires_grad_(True)
        fog = snow_creator(flake_intensities, kernels, 224, gpu_id=gpu_id)
    try:
        fog_im = transforms.ToPILImage()(fog[0, 0, :]).convert('RGB')
        fog_L = transforms.ToPILImage()(fog[0, 0, :]).convert('L')
        # fog_im.save(os.path.join(SAVE_DIR, 'fog.png'))
        fog_mask = 255 - np.array(fog_im)
        fog_mask = Image.fromarray(fog_mask).convert('RGB')
        # fog_mask.save("eval_result_test/fog.png")
        # foreground = fog_mask[:, :, ::-1].copy()
        alphamask = np.array(fog_im) / 255
        alphamask_inverse = 1.0 - alphamask
        foreground = np.ones((224, 224, 3), dtype=float) * 255
        background = np.array(pixel_img)
        background = background.astype(float)
        alphamask_inverse = alphamask_inverse.astype(float)
        if len(background.shape) == 2:
            background = np.dstack((background, background, background))
        if len(alphamask_inverse.shape) == 2:
            alphamask_inverse = np.dstack((alphamask_inverse, alphamask_inverse, alphamask_inverse))

        foreground = cv2.multiply(alphamask, foreground)
        background = cv2.multiply(alphamask_inverse, background)
        # Add the masked foreground and background
        outImage = cv2.add(foreground, background)
        outImage = Image.fromarray(outImage.astype(np.uint8))
        # outImage.save("eval_result_test/fogged.png")
    except:
        return pixel_img
    return outImage
