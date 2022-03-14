import transforms as T
from skimage.measure import compare_ssim
import h5py
from subsample import MaskFunc
import numpy as np
import torch
from matplotlib import pyplot as plt

def show_slices(data, slice_nums, cmap=None): # visualisation
    fig = plt.figure(figsize=(15,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')

def array_to_absImage(volume_kspace):
    volume_kspace2 = T.to_tensor(volume_kspace)     # Convert from numpy array to pytorch tensor
    return tensor_to_absImage(volume_kspace2)

def tensor_to_absImage(kspace):
    image = T.ifft2(kspace)                         # Apply Inverse Fourier Transform to get the complex image
    image_abs = T.complex_abs(image)                # Compute absolute value to get a real image
    return image_abs

def crop_image_320(image_abs):
    return T.center_crop(image_abs, [320, 320])

def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(gt.transpose(1, 2, 0),
                        pred.transpose(1, 2, 0),
                        multichannel=True,
                        data_range=gt.max())


def get_epoch_batch(subject_id, acc, center_fract, use_seed=True):
    ''' random select a few slices (batch_size) from each volume'''

    fname, rawdata_name, slice = subject_id

    with h5py.File(rawdata_name, 'r') as data:
        rawdata = data['kspace'][slice]

    slice_kspace = T.to_tensor(rawdata).unsqueeze(0)
    S, Ny, Nx, ps = slice_kspace.shape

    # apply random mask
    shape = np.array(slice_kspace.shape)
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    seed = None if not use_seed else tuple(map(ord, fname))
    mask = mask_func(shape, seed)

    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), slice_kspace)
    masks = mask.repeat(S, Ny, 1, ps)

    img_gt, img_und = T.ifft2(slice_kspace), T.ifft2(masked_kspace)

    # perform data normalization which is important for network to learn useful features
    # during inference there is no ground truth image so use the zero-filled recon to normalize
    norm = T.complex_abs(img_und).max()
    if norm < 1e-6: norm = 1e-6

    # normalized data
    img_gt, img_und, rawdata_und = img_gt / norm, img_und / norm, masked_kspace / norm

    return img_gt.squeeze(0), img_und.squeeze(0), rawdata_und.squeeze(0), masks.squeeze(0), norm