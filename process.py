import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from edsr import MultiNetwork as EDSR


def get_network_config():
    return {4: {'block': 8, 'feature': 48},
            3: {'block': 8, 'feature': 42},
            2: {'block': 8, 'feature': 26},
            1: {'block': 1, 'feature': 26}}


def load_model(model_path: str, device: str = 'cuda'):
    model = EDSR(get_network_config())
    model = model.to(torch.device(device))
    if not Path(model_path).exists():
        print('model does not exist')
        exit(-1)

    model.load_state_dict(torch.load(model_path))
    return model


def training(model, loader, scale, epoch):
    model.set_target_scale(scale)

    model.train()
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.networks[scale - 1].parameters(), lr=1e-4, weight_decay=0)

    for _ in range(epoch):
        for iteration, data in enumerate(loader, 1):
            input_tensor, target_tensor = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            output_tensor = model(input_tensor)
            loss_fn(output_tensor, target_tensor).backward()
            optimizer.step()


def inference(model, frame, scale):
    model.set_target_scale(scale)
    model.eval()

    with torch.no_grad():
        input_tensor_ = torch.from_numpy(frame).byte().cuda()
        input_tensor_ = input_tensor_.permute(2, 0, 1)
        input_tensor_ = input_tensor_.true_divide(255)
        input_tensor_.unsqueeze_(0)

        output_ = model(input_tensor_)
        output_ = output_.data[0].permute(1, 2, 0)
        output_ = output_ * 255
        output_ = torch.clamp(output_, 0, 255)

    torch.cuda.synchronize()
    return output_.cpu().numpy().astype(np.uint8)
