import natsort
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, json


def load_data(data):
    """Loads the saved numpy data
    Input Args:
        data (str) : path for data which has to loaded
    """
    assert data is not None, "Please provide data path"
    return np.load(file=data, allow_pickle=True)


def pad_input(input_data, req_shape=None, pow_two=False, multiple=4):
    """Pads the input data to the shape nearest to the
        power of 2 or multiples if specified.
    Input Args:
        input_data (ndarray) : Array of any dimension
        req_shape (tuple, optional) : Required shape of output
                                        if provided, output may not be
                                        closest power of 2
        If input data is of multilabel type, label type axis is ignored
        pow_two (bool): If input has to be padded to powers of 2
                        (overrides specified multiple value)
                        If False, input will be padded to nearest multiple
        multiple (int): Int specifying padding size
    Returns:
        padded_output (ndarray): Array where dimensions
        are specified by req_shape or are closest to the power of 2
        when req_shape is None
    """
    assert type(multiple) is None or int, "Invalid multiple type"
    assert pow_two or multiple, "Multiple not defined"
    assert type(input_data) is np.ndarray or torch.Tensor, "Input data format wrong"
    if len(input_data.shape) > 3:
        shape_of_input = input_data.shape[1:]
    else:
        shape_of_input = input_data.shape
    padding_list = []
    for idx, each_axis in enumerate(reversed(shape_of_input)):
        if req_shape is None:
            if pow_two:
                nearest_val = np.int_(2 ** (np.ceil(np.log(each_axis) / np.log(2))))
            if multiple:
                rem = each_axis % multiple
                nearest_val = each_axis + multiple - rem
        else:
            nearest_val = req_shape[::-1][idx]
        padding_list += [0, nearest_val - each_axis]

    if type(input_data) is np.ndarray:
        input_data = torch.from_numpy(input_data).float()
    padded_data = F.pad(input_data, tuple(padding_list), "constant", 0)
    return padded_data


def load_json_settings(json_file):
    """Loads and returns the json file with the
    training parameters
    """
    with open(json_file) as json_file:
        params = json.load(json_file)
    return params


json_file = "params/training_settings.json"
params = load_json_settings(json_file)
home_dir = params["params"]["home_dir"]


class CubeECHADataset(Dataset):
    """
    Dataset class to parse cube files to loader
    Input:
        eden_path: path to edensity cubes
        eneg_path: path to electroneg cubes
        mode: either 'train', 'test' or 'val
        csv_file: path to csv containing names of cube files and labels
        transform: data transformations if needed
    """

    #
    def __init__(self, eden_path, eneg_path, mode, transform, csv_file):
        self.valid_extensions_saves = [".npy"]
        self.eneg_path = eneg_path
        self.eden_path = eden_path
        self.mode = mode
        self.transform = transform
        self.csv_file = pd.read_csv(csv_file)

        if mode.lower() == "train":
            self.voxels_a = load_data(
                os.path.join(home_dir, "train_vox_info_ECHA_al_Eden.npy")
            )  #
            self.voxels_pr = load_data(
                os.path.join(home_dir, "train_vox_info_ECHA_pr_Eden.npy")
            )
            self.voxels = np.vstack([self.voxels_a, self.voxels_pr])
            self.train_files = self.csv_file
        if mode.lower() == "test":
            self.voxels_a = load_data(
                os.path.join(home_dir, "test_vox_info_ECHA_al_Eden.npy")
            )
            self.voxels_pr = load_data(
                os.path.join(home_dir, "test_vox_info_ECHA_pr_Eden.npy")
            )
            self.voxels = np.vstack([self.voxels_a, self.voxels_pr])
            self.test_files = self.csv_file
        if mode.lower() == "val":
            self.voxels_a = load_data(
                os.path.join(home_dir, "val_vox_info_ECHA_al_Eden.npy")
            )  #
            self.voxels_pr = load_data(
                os.path.join(home_dir, "val_vox_info_ECHA_pr_Eden.npy")
            )
            self.voxels = np.vstack([self.voxels_a, self.voxels_pr])
            self.val_files = self.csv_file

    def __len__(self):
        return len(self.csv_file)

    @property
    def getidx__(self):
        """returns index of file for identification purpose"""
        return self.idx

    def __getitem__(self, idx):
        """
        For a given index of file from the batch, the following processes are performed:
        1. Class label is firstly identified, either as prohibited or allowed from the csv_file parameter
        2. Based on this identifier, all files from the corresponding folders are loaded, e.g. prohibited eDen files
        from the prohibited eDen folder and corresponding ENeg files
        3. Thresholds for the loaded ENeg files are calculated using their percentile values and a mask is created
        4. Data is reshaped into the batch shape, padded if needed
        5. Reshaped eDen cube, thresholded mask and class type are returned in each batch
        """
        self.idx = idx
        data = self.csv_file.iloc[idx]
        self.class_type = torch.tensor(data[2])
        if self.class_type == 1:  # should be originally 1
            identifier = "allowed"
        else:
            identifier = "prohibited"

        eneg_path_gen = os.path.join(
            os.path.join(
                os.path.join(os.path.join(self.eneg_path, identifier), "eNeg"),
                self.mode.title(),
            ),
            self.mode + "_data",
        )
        eden_path_gen = os.path.join(
            os.path.join(
                os.path.join(os.path.join(self.eden_path, identifier), "eDen"),
                self.mode.title(),
            ),
            self.mode + "_data",
        )
        self.eneg = load_data(os.path.join(eneg_path_gen, data[1]))
        self.cube_vol = np.array(load_data(os.path.join(eden_path_gen, data[0])))

        shapes1 = list(self.voxels[idx, :])
        shapes1.append(3)

        low_10 = np.percentile(self.eneg, 10)
        upper_10 = np.percentile(self.eneg, 90)
        mask = np.zeros(self.eneg.shape)
        mask[self.eneg < low_10] = 1
        mask[self.eneg > upper_10] = 2
        mask = [int(a) for a in mask]

        mask = np.squeeze(np.eye(3)[np.array(mask).reshape(-1)])

        shapes = list(self.voxels[idx, :])
        shapes.append(3)
        mask = mask.reshape(tuple(shapes))
        mask = np.moveaxis(mask, -1, 0)
        mask = np.moveaxis(mask, -1, 1)

        self.cube_vol = self.cube_vol.reshape(self.voxels[idx, :])
        self.cube_vol = np.moveaxis(self.cube_vol, -1, 0)
        self.cube_vol = np.expand_dims(self.cube_vol, axis=0)
        mask = pad_input(mask, req_shape=None)
        self.cube_vol = pad_input(self.cube_vol, req_shape=None)

        if self.transform:
            self.cube_vol = self.transform(self.cube_vol)
            mask = self.transform(mask)

        return self.cube_vol, mask, self.class_type, idx
