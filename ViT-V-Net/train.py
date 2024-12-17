from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from natsort import natsorted
from device_helper import device
import time
from torch.utils.data import Dataset
import nibabel as nib

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

class LungCTRegistrationDataset(Dataset):
    def __init__(self, image_dir, transform=None, split="train"):
        """
        Dataset for loading lung CT registration data.

        Args:
            image_dir (str): Directory containing the images.
            transform (callable, optional): A function/transform to apply to the images.
            split (str): 'train' or 'val' to specify dataset split.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load all files and filter by split
        all_fbct_files = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')))
        all_cbct1_files = sorted(glob.glob(os.path.join(image_dir, '*_0001.nii.gz')))
        all_cbct2_files = sorted(glob.glob(os.path.join(image_dir, '*_0002.nii.gz')))

        if split == "train":
            # First eleven patients
            self.fbct_files = [f for f in all_fbct_files if int(os.path.basename(f).split('_')[0]) <= 10]
            self.cbct1_files = [f for f in all_cbct1_files if int(os.path.basename(f).split('_')[0]) <= 10]
            self.cbct2_files = [f for f in all_cbct2_files if int(os.path.basename(f).split('_')[0]) <= 10]
        elif split == "val":
            # Last three patients
            self.fbct_files = [f for f in all_fbct_files if int(os.path.basename(f).split('_')[0]) >= 11]
            self.cbct1_files = [f for f in all_cbct1_files if int(os.path.basename(f).split('_')[0]) >= 11]
            self.cbct2_files = [f for f in all_cbct2_files if int(os.path.basename(f).split('_')[0]) >= 11]

        assert len(self.fbct_files) == len(self.cbct1_files) == len(self.cbct2_files), \
            "Mismatch in the number of FBCT and CBCT images for the chosen split."

    def __len__(self):
        """
        Returns the total number of pairs (FBCT-CBCT1, FBCT-CBCT2) in the dataset.
        """
        return len(self.fbct_files) * 2  # Two pairs per patient

    def __getitem__(self, idx):
        patient_idx = idx // 2
        pair_type = idx % 2

        # Load FBCT and CBCT files
        fbct = nib.load(self.fbct_files[patient_idx]).get_fdata(dtype=np.float32)
        cbct_path = self.cbct1_files[patient_idx] if pair_type == 0 else self.cbct2_files[patient_idx]
        cbct = nib.load(cbct_path).get_fdata(dtype=np.float32)

        # Add channel and batch axes if needed
        if len(fbct.shape) == 3:
            fbct = fbct[np.newaxis, np.newaxis, ...]  # Add [Batch=1, Channel=1] axes
        if len(cbct.shape) == 3:
            cbct = cbct[np.newaxis, np.newaxis, ...]

        # Apply transformations (if any)
        if self.transform:
            fbct = self.transform(fbct)
            cbct = self.transform(cbct)

        # Flip along depth axis if specified in the transformation
        axis_to_flip = 2  # Depth (Z-axis)
        fbct = np.flip(fbct, axis=axis_to_flip).copy()
        cbct = np.flip(cbct, axis=axis_to_flip).copy()
        #print(f"FBCT shape after .copy(): {fbct.shape}")

        # Convert to PyTorch tensors
        fbct = torch.tensor(fbct, dtype=torch.float32).squeeze(0)
        cbct = torch.tensor(cbct, dtype=torch.float32).squeeze(0)

        return fbct, cbct, pair_type
def main():
    # Set device using the helper function
    compute_device = device()
    print(f"Using device: {compute_device}")
    dev = device()
    batch_size = 1
    if dev == "gpu":
        batch_size = 2
    print("Device and batch_size: ", dev, batch_size)

    train_dir = "/app/Release_06_12_23/imagesTr"  # Ensure paths are correct
    val_dir = "/app/Release_06_12_23/imagesTr_Val"
    save_dir = '/app/Models/Vitvnet_1'
    lr = 0.0001
    epoch_start = 0
    max_epoch = 100  ##

    # Timer for main function
    main_start_time = time.time()

    # Model configuration
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    """reg_model = utils.register_model((160, 192, 224), 'nearest').to(compute_device)
    model = models.ViTVNet(config_vit, img_size=(160, 192, 224)).to(compute_device)"""  #old ViT-V-Net data sizes
    reg_model = utils.register_model((256, 192, 192), 'nearest').to(compute_device)
    model = models.ViTVNet(config_vit, img_size=(256, 192, 192)).to(compute_device)

    train_composed = transforms.Compose([
        trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])

    val_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])

    # Dataset and DataLoader setup
    train_set = LungCTRegistrationDataset(train_dir, transform=train_composed, split="train")
    val_set = LungCTRegistrationDataset(val_dir, transform=val_composed, split="val")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Optimizer and loss configuration
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterions = [nn.MSELoss(), losses.Grad3d(penalty='l2')]
    weights = [1, 0.02]

    
    print(f"Training for {max_epoch} epoch(s)")
    for epoch in range(epoch_start, max_epoch):
        print(f"Epoch {epoch + 1}/{max_epoch} starting...")
        loss_all = AverageMeter()

        for idx, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            # Data preparation
            fbct, cbct, pair_type = data
            #print("FBCT shape:", fbct.shape)
            #print("CBCT shape:", cbct.shape)
            fbct, cbct = fbct.to(compute_device), cbct.to(compute_device)
            
            # Forward pass
            x_in = torch.cat((fbct, cbct), dim=1)
            output = model(x_in)

            # Loss computation
            loss = sum(
                criterions[i](output[i], cbct) * weights[i]
                for i in range(len(criterions))
            )
            loss_all.update(loss.item(), cbct.numel())

            # Backpropagation
            loss.backward()
            optimizer.step()

            print(f"Iteration {idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} completed. Average Loss: {loss_all.avg:.4f}")

    # End timer for main
    main_end_time = time.time()
    elapsed_main_time = main_end_time - main_start_time
    print(f"Main function runtime: {elapsed_main_time:.2f} seconds")


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    main()