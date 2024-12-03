import glob
from torch.utils.tensorboard import SummaryWriter
import logging
import os, losses, utils, nrrd
import shutil
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from torchsummary import summary
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from device_helper import device
import pickle
from utils import dice_val, save_nifti, register_model  # Add your utility functions as needed


def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], linewidth=0.8, **kwargs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))

def load_subject_data(test_dir, subject_index=0):
    """
    Load the subject image and segmentation from the test directory.

    Args:
        test_dir (str): Path to the test directory containing subject .pkl files.
        subject_index (int): Index of the subject to load.

    Returns:
        tuple: Loaded subject image and segmentation as PyTorch tensors.
    """
    test_files = natsorted(glob.glob(f"{test_dir}/*.pkl"))
    if not test_files:
        raise FileNotFoundError(f"No .pkl files found in {test_dir}")

    # Select the desired subject file
    subject_file = test_files[subject_index]
    print(f"Loading subject data from: {subject_file}")

    # Load the subject file
    with open(subject_file, 'rb') as f:
        subject_data = pickle.load(f)

    # Assuming the subject file contains a tuple: (image, segmentation)
    subject_image = subject_data[0]
    subject_segmentation = subject_data[1]

    # Convert to PyTorch tensors
    subject_image_tensor = torch.tensor(subject_image).unsqueeze(0).unsqueeze(0).to(device())
    subject_segmentation_tensor = torch.tensor(subject_segmentation).unsqueeze(0).unsqueeze(0).to(device())

    return subject_image_tensor, subject_segmentation_tensor

def visualize_image_and_segmentation(image, segmentation, axis=0, slice_idx=50):
    """
    Visualizes a slice of the 3D image and segmentation.

    Args:
        image (torch.Tensor or np.ndarray): The aligned image (C, D, H, W).
        segmentation (torch.Tensor or np.ndarray): The aligned segmentation (C, D, H, W).
        axis (int): Axis along which to slice the 3D data (0 for axial, 1 for sagittal, 2 for coronal).
        slice_idx (int): Index of the slice to visualize.
    """
    # Convert image and segmentation to numpy if they are in tensor form
    image = image.squeeze().cpu().detach().numpy()
    segmentation = segmentation.squeeze().cpu().detach().numpy()

    # Slice the image and segmentation along the specified axis
    if axis == 0:  # Axial (D axis)
        image_slice = image[slice_idx, :, :]
        segmentation_slice = segmentation[slice_idx, :, :]
    elif axis == 1:  # Sagittal (H axis)
        image_slice = image[:, slice_idx, :]
        segmentation_slice = segmentation[:, slice_idx, :]
    elif axis == 2:  # Coronal (W axis)
        image_slice = image[:, :, slice_idx]
        segmentation_slice = segmentation[:, :, slice_idx]
    else:
        raise ValueError("Axis must be 0 (axial), 1 (sagittal), or 2 (coronal)")

    # Plot the image and segmentation
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image_slice, cmap='gray')
    ax[0].set_title('Aligned Image Slice')
    ax[0].axis('off')

    ax[1].imshow(segmentation_slice, cmap='jet')
    ax[1].set_title('Aligned Segmentation Slice')
    ax[1].axis('off')

    plt.show()

# Function to load the atlas image from a pickle file

def load_atlas(atlas_path):
    with open(atlas_path, 'rb') as f:
        atlas_data = pickle.load(f)
    return atlas_data[0]  # Assuming the atlas image is stored under the 'image' key in the pickle


def visualize_comparison(atlas_path, subject_image, aligned_image, axis=0, slice_idx=50):
    """
    Visualizes the atlas image, subject image, and aligned image on the same plot for comparison.

    Args:
        atlas_path (str): Path to the atlas image pickle file.
        subject_image (torch.Tensor or np.ndarray): The subject image (C, D, H, W).
        aligned_image (torch.Tensor or np.ndarray): The aligned subject image (C, D, H, W).
        axis (int): Axis along which to slice the 3D data (0 for axial, 1 for sagittal, 2 for coronal).
        slice_idx (int): Index of the slice to visualize.
    """
    # Load the atlas image
    atlas_image= load_atlas(atlas_path)

    # Convert tensors to numpy arrays (if they are tensors)
    subject_image = subject_image.squeeze().cpu().detach().numpy()
    aligned_image = aligned_image.squeeze().cpu().detach().numpy()

    # Slice the images along the specified axis
    if axis == 0:  # Axial (D axis)
        atlas_slice = atlas_image[slice_idx, :, :]
        subject_slice = subject_image[slice_idx, :, :]
        aligned_slice = aligned_image[slice_idx, :, :]
    elif axis == 1:  # Sagittal (H axis)
        atlas_slice = atlas_image[:, slice_idx, :]
        subject_slice = subject_image[:, slice_idx, :]
        aligned_slice = aligned_image[:, slice_idx, :]
    elif axis == 2:  # Coronal (W axis)
        atlas_slice = atlas_image[:, :, slice_idx]
        subject_slice = subject_image[:, :, slice_idx]
        aligned_slice = aligned_image[:, :, slice_idx]
    else:
        raise ValueError("Axis must be 0 (axial), 1 (sagittal), or 2 (coronal)")

    # Plot the atlas image, subject image, and aligned image
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Atlas Image
    ax[0].imshow(atlas_slice, cmap='gray')
    ax[0].set_title('Atlas Image Slice')
    ax[0].axis('off')

    # Subject Image (Original)
    ax[1].imshow(subject_slice, cmap='gray')
    ax[1].set_title('Subject Image Slice')
    ax[1].axis('off')

    # Aligned Image
    ax[2].imshow(aligned_slice, cmap='gray')
    ax[2].set_title('Aligned Image Slice')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()



def dice_coefficient(pred, truth):
    """
    Computes the Dice Similarity Coefficient (DSC) between the predicted and true segmentation.

    Args:
        pred (np.ndarray or torch.Tensor): Predicted segmentation (binary).
        truth (np.ndarray or torch.Tensor): Ground truth segmentation (binary).

    Returns:
        float: The DSC value.
    """
    # Ensure that the inputs are binary (0 or 1)
    pred = pred.detach().cpu().numpy().astype(bool)
    truth = truth.detach().cpu().numpy().astype(bool)

    # Compute intersection and union
    intersection = np.sum(pred & truth)
    union = np.sum(pred) + np.sum(truth)

    # Compute DSC
    dsc = 2 * intersection / union
    return dsc


def main(subject_image, subject_segmentation, atlas_path):
    """
    Aligns a subject's image and segmentation with an atlas using a pre-trained model.

    Args:
        subject_image (torch.Tensor): Subject's image tensor (C, D, H, W).
        subject_segmentation (torch.Tensor): Subject's segmentation tensor (C, D, H, W).
        atlas_path (str): Path to the atlas file, expected to be a pickle file containing a tuple or dictionary.

    Returns:
        tuple: Deformed subject image and segmentation tensors aligned with the atlas.
    """
    try:
        # Load the atlas file
        with open(atlas_path, 'rb') as f:
            atlas_data = pickle.load(f)

        # Handle tuple structure
        if isinstance(atlas_data, tuple):
            print("Atlas file contains a tuple.")
            atlas_image = atlas_data[0]
            atlas_segmentation = atlas_data[1]

        # Handle dictionary structure
        elif isinstance(atlas_data, dict):
            print("Atlas file contains a dictionary.")
            atlas_image = atlas_data['image']
            atlas_segmentation = atlas_data['segmentation']

        else:
            raise ValueError(f"Unsupported data format in atlas file: {type(atlas_data)}")

        # Convert atlas data to PyTorch tensors
        atlas_image_tensor = torch.tensor(atlas_image).unsqueeze(0).unsqueeze(0).to(device())
        atlas_segmentation_tensor = torch.tensor(atlas_segmentation).unsqueeze(0).unsqueeze(0).to(device())

        # Load the pre-trained model
        config_vit = CONFIGS_ViT_seg['ViT-V-Net']
        model = models.ViTVNet(config_vit, img_size=(160, 192, 224))
        best_model = torch.load('experiments/pretrained_models/ViTVNet_Validation_dsc0.726.pth.tar', map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(best_model)
        model.to(device())
        model.eval()

        # Register model for deformation
        reg_model = register_model((160, 192, 224), 'nearest')
        reg_model.to(device())

        # Prepare inputs for the model
        x_in = torch.cat((subject_image, atlas_image_tensor), dim=1)
        x_def, flow = model(x_in)
        def_segmentation = reg_model([subject_segmentation.float().to(device()), flow])

        print("Alignment with atlas complete.")
        return x_def.squeeze(0), def_segmentation.squeeze(0)

    except FileNotFoundError:
        print(f"File not found: {atlas_path}")
        raise
    except KeyError as e:
        print(f"KeyError: Missing key in atlas file: {e}")
        raise
    except Exception as e:
        print(f"Error during alignment: {e}")
        raise



def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


if __name__ == '__main__':
    test_dir = 'IXI_data/Test'
    subject_image, subject_segmentation = load_subject_data(test_dir, subject_index=0)
    atlas_path = 'IXI_data/atlas.pkl'
    aligned_image, aligned_segmentation = main(subject_image, subject_segmentation, atlas_path)
    print(f"Aligned image shape: {aligned_image.shape}")
    print(f"Aligned segmentation shape: {aligned_segmentation.shape}")
 
    # Visualize the aligned images and segmentations
    #visualize_image_and_segmentation(aligned_image, aligned_segmentation, axis=0, slice_idx=50)
    #visualize_image_and_segmentation(subject_image, subject_segmentation, axis=0, slice_idx=50)
    visualize_comparison(atlas_path, subject_image, aligned_image, axis=0, slice_idx=50)

    # Compute and print DSC for a specific slice
    dsc_value = dice_coefficient(aligned_segmentation, subject_segmentation)
    print(f"Dice Similarity Coefficient: {dsc_value:.4f}")
