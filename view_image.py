import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Function to visualize slices from a 3D volume
def display_slices(volume, title_prefix):
    """
    Display axial, coronal, and sagittal slices from a 3D volume.

    Args:
        volume (numpy.ndarray or torch.Tensor): 3D volume of shape (D, H, W).
        title_prefix (str): Prefix for the plot titles (e.g., FBCT, CBCT1).
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Extract central slices along each axis
    axial_slice = volume[volume.shape[0] // 2, :, :]  # Axial plane (Z-axis)
    coronal_slice = volume[:, volume.shape[1] // 2, :]  # Coronal plane (Y-axis)
    sagittal_slice = volume[:, :, volume.shape[2] // 2]  # Sagittal plane (X-axis)

    # Plot the slices
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(axial_slice, cmap="gray")
    plt.title(f"{title_prefix} - Axial")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(coronal_slice, cmap="gray")
    plt.title(f"{title_prefix} - Coronal")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sagittal_slice, cmap="gray")
    plt.title(f"{title_prefix} - Sagittal")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Dummy 3D volumes (replace with actual CT volumes)
    FBCT = nib.load(r"C:\Users\frani\Desktop\AMS_izziv_trainingCT\Release_06_12_23\imagesTr\ThoraxCBCT_0010_0000.nii.gz") 
    CBCT1 = nib.load(r"C:\Users\frani\Desktop\AMS_izziv_trainingCT\Release_06_12_23\imagesTr\ThoraxCBCT_0010_0000.nii.gz")  
    CBCT2 = nib.load(r"C:\Users\frani\Desktop\AMS_izziv_trainingCT\Release_06_12_23\imagesTr\ThoraxCBCT_0010_0000.nii.gz") 
    print(CBCT1.shape)
    FBCT = np.asarray(FBCT.get_fdata())
    CBCT1 = np.asarray(CBCT1.get_fdata())
    CBCT2 = np.asarray(CBCT2.get_fdata())
    print(CBCT1.shape)
    # Display slices for each volume
    display_slices(FBCT, "FBCT")
    display_slices(CBCT1, "CBCT1")
    display_slices(CBCT2, "CBCT2")
