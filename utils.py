import numpy as np
from PIL import Image
import torch

def image_from_path(image_path):
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()

    return data_lowlight


def scale_image(data_lowlight, scale_factor, device):
    h = ((data_lowlight.shape[0]) // scale_factor) * scale_factor
    w = ((data_lowlight.shape[1]) // scale_factor) * scale_factor
    # print("cropped height is ", h)
    # print("cropped width is", w)
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    return data_lowlight


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
