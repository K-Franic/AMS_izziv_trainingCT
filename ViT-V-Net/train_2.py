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
import wandb
from sklearn.metrics import mean_absolute_error

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
    def __init__(self, image_dir, mask_dir, transform=None, split="train"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Load all image files
        all_fbct_files = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')))
        all_cbct1_files = sorted(glob.glob(os.path.join(image_dir, '*_0001.nii.gz')))
        all_cbct2_files = sorted(glob.glob(os.path.join(image_dir, '*_0002.nii.gz')))

        def extract_patient_number(filename):
            """Extracts the zero-padded patient number from the filename."""
            basename = os.path.basename(filename)
            patient_number_str = basename.split('_')[1]  # Extract patient number
            return int(patient_number_str)  # Convert to integer for comparison

        def format_patient_number(number):
            """Formats the patient number with zero-padding."""
            return f"{number:04}"  # Zero-pad to 4 digits

        # Split images by training/validation sets
        if split == "train":
            self.fbct_files = [f for f in all_fbct_files if extract_patient_number(f) <= 10]
            self.cbct1_files = [f for f in all_cbct1_files if extract_patient_number(f) <= 10]
            self.cbct2_files = [f for f in all_cbct2_files if extract_patient_number(f) <= 10]
        elif split == "val":
            self.fbct_files = [f for f in all_fbct_files if extract_patient_number(f) >= 11]
            self.cbct1_files = [f for f in all_cbct1_files if extract_patient_number(f) >= 11]
            self.cbct2_files = [f for f in all_cbct2_files if extract_patient_number(f) >= 11]

        # Ensure the number of images is consistent across types
        assert len(self.fbct_files) == len(self.cbct1_files) == len(self.cbct2_files), \
            "Mismatch in the number of FBCT, CBCT1, and CBCT2 images for the chosen split."

        # Match masks to images, zero-padding patient numbers
        self.fbct_masks = [os.path.join(mask_dir, f"ThoraxCBCT_{format_patient_number(extract_patient_number(f))}_0000.nii.gz") for f in self.fbct_files]
        self.cbct1_masks = [os.path.join(mask_dir, f"ThoraxCBCT_{format_patient_number(extract_patient_number(f))}_0001.nii.gz") for f in self.cbct1_files]
        self.cbct2_masks = [os.path.join(mask_dir, f"ThoraxCBCT_{format_patient_number(extract_patient_number(f))}_0002.nii.gz") for f in self.cbct2_files]

        # Assert that the number of masks matches the number of images
        all_masks = self.fbct_masks + self.cbct1_masks + self.cbct2_masks
        assert len(all_masks) == len(self.fbct_files) + len(self.cbct1_files) + len(self.cbct2_files), \
            "Mismatch between the total number of images and masks."

    def __len__(self):
        return len(self.fbct_files) + len(self.cbct1_files) + len(self.cbct2_files)

    def __getitem__(self, idx):
        if idx < len(self.fbct_files):
            image_path = self.fbct_files[idx]
            mask_path = self.fbct_masks[idx]
        elif idx < len(self.fbct_files) + len(self.cbct1_files):
            image_path = self.cbct1_files[idx - len(self.fbct_files)]
            mask_path = self.cbct1_masks[idx - len(self.fbct_files)]
        else:
            image_path = self.cbct2_files[idx - len(self.fbct_files) - len(self.cbct1_files)]
            mask_path = self.cbct2_masks[idx - len(self.fbct_files) - len(self.cbct1_files)]

        image = nib.load(image_path).get_fdata(dtype=np.float32)
        mask = nib.load(mask_path).get_fdata(dtype=np.float32)
        #mask = mask.astype(np.int16)

        if len(image.shape) == 3:
            image = image[np.newaxis, np.newaxis, ...]
        if len(mask.shape) == 3:
            mask = mask[np.newaxis, np.newaxis, ...]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = torch.tensor(image, dtype=torch.float32).squeeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).squeeze(0)

        return image, mask
                      
def main():
    # Initialize WandB
    wandb.init(
        project="KF-ct-registration_test",  # Your project name on WandB
        config={
            "learning_rate": 0.0001,
            "epochs": 200,
            "batch_size": 1,
            "optimizer": "Adam",
            "model": "ViT-V-Net"
        }
    )
    config = wandb.config

    # Set device using the helper function
    compute_device = device()
    print(f"Using device: {compute_device}")
    dev = device()
    batch_size = 1
    if dev == "cuda":
        batch_size = 1
    print("Device and batch_size: ", dev, batch_size)

    train_dir = "/app/Release_06_12_23/imagesTr"
    val_dir = "/app/Release_06_12_23/imagesTr"  # Same directory for validation
    mask_dir = "/app/Release_06_12_23/masksTr"  # Mask directory
    save_dir = '/app/ViT-V-Net/Models/Vitvnet200'
    lr = config.learning_rate
    epoch_start = 0
    max_epoch = config.epochs

    # Model and Registration Setup
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    reg_model = utils.register_model((256, 192, 192), 'nearest').to(compute_device)
    model = models.ViTVNet(config_vit, img_size=(256, 192, 192)).to(compute_device)
    updated_lr = lr

    # Data Augmentation
    train_composed = transforms.Compose([
        trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])
    val_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])

    # Datasets and DataLoaders
    train_set = LungCTRegistrationDataset(train_dir, mask_dir, transform=train_composed, split="train")
    val_set = LungCTRegistrationDataset(val_dir, mask_dir, transform=val_composed, split="val")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterions = [nn.MSELoss(), losses.Grad3d(penalty='l2')]
    weights = [1, 0.02]

    # Training Loop
    best_metric = 0

    for epoch in range(epoch_start, max_epoch):
        print(f"Epoch {epoch + 1}/{max_epoch}")
        model.train()
        loss_all = AverageMeter()

        for idx, data in enumerate(train_loader):
            # Training Step
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x, y = data[0], data[1]

            # Forward Pass
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            loss = sum(criterions[i](output[i], y) * weights[i] for i in range(len(criterions)))

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Metrics
            loss_all.update(loss.item(), y.numel())

            # Log training metrics to WandB
            wandb.log({"Train Loss": loss.item(), "Epoch": epoch + 1})

        print(f"Training Loss: {loss_all.avg:.4f}")

        #torch.cuda.empty_cache()  # Free up unused GPU memory
        # Validation
        eval_metric = validate(model, reg_model, val_loader, device="cuda")
        best_metric = max(eval_metric, best_metric)

        # Log validation metrics to WandB
        wandb.log({"Validation Metric": eval_metric, "Best Metric": best_metric, "Epoch": epoch + 1})

        # Save model checkpoint to WandB
        checkpoint_path = f"{save_dir}/metric_{eval_metric:.3f}.pth.tar"
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_metric': best_metric,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename=f'metric_{eval_metric:.3f}.pth.tar')
        wandb.save(checkpoint_path)

    # Finish WandB run
    wandb.finish()

# Validation Function
"""def validate(model, reg_model, val_loader, device):
    model.eval()
    eval_metric = AverageMeter()

    with torch.no_grad():
        for data in val_loader:
            data = [t.cuda() for t in data]
            x, y, x_seg, y_seg = data
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            def_out = reg_model([x_seg.float(), output[1]])
            metric = utils.dice_val(def_out.long(), y_seg.long(), num_classes=46)
            eval_metric.update(metric.item(), x.size(0))

    return eval_metric.avg
# Add additional imports if needed for new metrics
from sklearn.metrics import mean_absolute_error
"""
# Updated validate function
def validate(model, reg_model, val_loader, device):
    model.eval()
    dice_meter = AverageMeter()
    mse_meter = AverageMeter()
    mae_meter = AverageMeter()  # Example: Mean Absolute Error

    with torch.no_grad():
        for data in val_loader:
            data = [t.to(device) for t in data]
            x, y = data[0], data[1]  # Image and corresponding mask

            # Forward Pass
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)

            # Compute Dice, MSE, MAE using segmentation masks
            def_out = reg_model([x, output[1]])

            # Compute Dice
            dice = utils.dice_val(def_out.long(), y.long())
            dice_meter.update(dice.item(), x.size(0))

            # Compute MSE
            mse = MSE_torch(def_out, y)
            mse_meter.update(mse.item(), x.size(0))

            # Compute MAE (if applicable)
            mae = mean_absolute_error(def_out.cpu().numpy().flatten(), y.cpu().numpy().flatten())
            mae_meter.update(mae, x.size(0))

    print(f"Validation Results - Dice: {dice_meter.avg:.4f}, MSE: {mse_meter.avg:.4f}, MAE: {mae_meter.avg:.4f}")

    # Log metrics to WandB
    wandb.log({
        "Validation Dice": dice_meter.avg,
        "Validation MSE": mse_meter.avg,
        "Validation MAE": mae_meter.avg,
    })

    # Return main metric for checkpointing
    return dice_meter.avg

# Checkpoint Saver
def save_checkpoint(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))

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
    torch.cuda.empty_cache()
    main()