import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import sys
import utils.utils as utils
from My_unet.unet import UNet

input_dir = Path('./predictions/input/')
output_dir = Path('./predictions/output/')


def pred_img(
        model,
        img,
        device,
        scale=1,
        out_treshold=0.5
):
    model.eval()
    preprocessed_img = torch.from_numpy(utils.BasicDataset.preprocess(None, img, scale, is_mask=False))
    preprocessed_img = preprocessed_img.unsqueeze(0)
    preprocessed_img = preprocessed_img.to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(preprocessed_img).cpu()
        output = F.interpolate(output, (img.size[1], img.size[0]), mode='bilinear')
        if model.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_treshold
    return mask[0].long().squeeze().numpy()


def get_output_filenames(args):
    def _generate_name(fn):
        fn = os.path.basename(fn)
        return f'{os.path.splitext(fn)[0]}_OUT.tif'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    pretrained_model_dir = Path('./saved_models/model9vd8zxb0 26.03.2023 loss0,1.pth')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    list_input = list(input_dir.glob('*.tif'))
    args = argparse.Namespace(input=list_input, output=None)
    out_files = get_output_filenames(args)
    # print(list_input, input_dir, out_files)
    model = UNet(n_channels=3, n_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')
    model.to(device=device)
    state_dict = torch.load(pretrained_model_dir, map_location=device)
    # print(state_dict)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)
    logging.info('Model loaded!')
    # data_list = []
    np.set_printoptions(threshold=sys.maxsize)
    for i, filename in enumerate(list_input):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = pred_img(model=model,
                        img=img,
                        device=device,
                        )
        print(mask)
        out_filename = out_files[i]
        result = mask_to_image(mask, mask_values)
        result.save(output_dir.joinpath(out_filename))
        logging.info(f'Mask saved to {out_filename}')
        logging.info(f'Visualizing results for image {filename}, close to continue...')
        utils.plot_img_and_mask(img, mask)


