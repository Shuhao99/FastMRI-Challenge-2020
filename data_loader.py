import h5py, os
from functions import transforms as T
from functions import provided_functions as P
from torch.utils.data import DataLoader
from functions.provided_functions import get_epoch_batch
import torch
from matplotlib import pyplot as plt

class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction, use_seed):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.use_seed = use_seed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(subject_id, self.acceleration, self.center_fraction, self.use_seed)


def load_data_path(train_data_path, val_data_path):
    """ Go through each subset (training, validation) and list all
    the file names, the file paths and the slices of subjects in the training and validation sets
    """

    data_list = {}
    train_and_val = ['train', 'val']
    data_path = [train_data_path, val_data_path]

    for i in range(len(data_path)):

        data_list[train_and_val[i]] = []

        which_data_path = data_path[i]

        for fname in sorted(os.listdir(which_data_path)):

            subject_data_path = os.path.join(which_data_path, fname)

            if not os.path.isfile(subject_data_path): continue

            with h5py.File(subject_data_path, 'r') as data:
                num_slice = data['kspace'].shape[0]

            # the first 5 slices are mostly noise so it is better to exlude them
            data_list[train_and_val[i]] += [(fname, subject_data_path, slice) for slice in range(5, num_slice)]

    return data_list


if __name__ == '__main__':

    data_path_train = 'train1'
    data_path_val = 'train2'
    data_list = load_data_path(data_path_train, data_path_val)  # first load all file names, paths and slices.

    acc = 8
    cen_fract = 0.04
    seed = False  # random masks for each slice
    num_workers = 12  # data loading is faster using a bigger number for num_workers. 0 means using one cpu to load data

    # create data loader for training set. It applies same to validation set as well
    train_dataset = MRIDataset(data_list['train'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=num_workers)
    for iteration, sample in enumerate(train_loader):

        img_gt, img_und, rawdata_und, masks, norm = sample

        # stack different slices into a volume for visualisation
        A = masks[..., 0].squeeze()
        B = torch.log(T.complex_abs(rawdata_und) + 1e-9).squeeze()
        C = T.complex_abs(img_und).squeeze()
        D = T.complex_abs(img_gt).squeeze()
        all_imgs = torch.stack([A, B, C, D], dim=0)

        # from left to right: mask, masked kspace, undersampled image, ground truth
        P.show_slices(all_imgs, [0, 1, 2, 3], cmap='gray')
        plt.pause(1)

        if iteration >= 3: break  # show 4 random slices