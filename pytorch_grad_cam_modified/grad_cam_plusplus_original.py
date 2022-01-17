import cv2
import numpy as np
import torch
from pytorch_grad_cam_modified.base_cam_original import BaseCAM

class GradCAMPlusPlus_original(BaseCAM):
    def __init__(self, model, target_layer, gpu_id=0,
        reshape_transform=None):
        super(GradCAMPlusPlus_original, self).__init__(model, target_layer, gpu_id, 
            reshape_transform)

    def get_cam_weights(self, input_tensor, 
                              target_category, 
                              activations, 
                              grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2*grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2*grads_power_2 + 
            sum_activations[:, :, None, None]*grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0)*aij
        weights = np.sum(weights, axis=(2, 3))
        return weights