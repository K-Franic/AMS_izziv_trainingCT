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
    def __init__(self, image_dir, transform=None, split="train"):
        """
        Dataset for loading lung CT registration data.

        Args:
            image_dir (str): Directory containing the images.
            transform (callable, optional): A function/transform to apply to the images.
            split (str): "train" or "val", determines which subset of patients to include.
        """
        self.image_dir = image_dir
        self.transform = transform

        # List all files
        self.fbct_files = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')))
        self.cbct1_files = sorted(glob.glob(os.path.join(image_dir, '*_0001.nii.gz')))
        self.cbct2_files = sorted(glob.glob(os.path.join(image_dir, '*_0002.nii.gz')))

        """# Extract patient IDs
        def get_patient_id(filepath):
            filename = os.path.basename(filepath)
            return int(filename.split('_')[1])
        
        # Filter by split
        if split == "train":
            self.fbct_files = [f for f in all_fbct_files if get_patient_id(f) <= 10]
            self.cbct1_files = [f for f in all_cbct1_files if get_patient_id(f) <= 10]
            self.cbct2_files = [f for f in all_cbct2_files if get_patient_id(f) <= 10]
        elif split == "val":
            self.fbct_files = [f for f in all_fbct_files if 11 <= get_patient_id(f) <= 13]
            self.cbct1_files = [f for f in all_cbct1_files if 11 <= get_patient_id(f) <= 13]
            self.cbct2_files = [f for f in all_cbct2_files if 11 <= get_patient_id(f) <= 13]
        else:
            raise ValueError("Split must be 'train' or 'val'.")"""

        # Check file counts
        assert len(self.fbct_files) == len(self.cbct1_files) == len(self.cbct2_files), \
            "Mismatch in the number of FBCT and CBCT images."

    def __len__(self):
        """Returns the total number of pairs (FBCT-CBCT1, FBCT-CBCT2) in the dataset."""
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

        # Convert to PyTorch tensors
        fbct = torch.tensor(fbct, dtype=torch.float32).squeeze(0)
        cbct = torch.tensor(cbct, dtype=torch.float32).squeeze(0)

        return fbct, cbct, pair_type

def main():
    # Initialize WandB
    wandb.init(
        project="KF-ct-registration_test",  # Your project name on WandB
        config={
            "learning_rate": 0.0001,
            "epochs": 1,
            "batch_size": 2,
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
        batch_size = 2
    print("Device and batch_size: ", dev, batch_size)

    train_dir = "/app/Release_06_12_23/imagesTr"
    val_dir = "/app/Release_06_12_23/imagesTr_val"
    save_dir = '/app/ViT-V-Net/Models/Vitvnet1'
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
    train_set = LungCTRegistrationDataset(train_dir, transform=train_composed, split="train")
    val_set = LungCTRegistrationDataset(val_dir, transform=val_composed, split="val")
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
            x, y, x_seg, y_seg = data
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            def_out = reg_model([x_seg.float(), output[1]])

            # Compute Dice
            dice = utils.dice_val(def_out.long(), y_seg.long(), num_classes=46)
            dice_meter.update(dice.item(), x.size(0))

            # Compute MSE
            mse = MSE_torch(def_out, y_seg)
            mse_meter.update(mse.item(), x.size(0))

            # Compute MAE (if applicable)
            mae = mean_absolute_error(def_out.cpu().numpy().flatten(), y_seg.cpu().numpy().flatten())
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
    main()