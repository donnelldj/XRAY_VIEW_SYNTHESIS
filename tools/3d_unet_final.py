import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os

# Define a custom dataset class to load the npz files directly
class DRRPairsDataset(Dataset):
    def __init__(self, npz_dir: Path):
        """
        Custom dataset class to load npz files from the specified directory.
        """
        self.npz_dir = npz_dir
        self.files = list(npz_dir.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No npz files found in directory: {npz_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_file = self.files[idx]
        data = np.load(npz_file)
        return {
            "bp": torch.tensor(data["ap"], dtype=torch.float32),  # Back-projected AP view
            "lat": torch.tensor(data["lat"], dtype=torch.float32),  # Lateral ground truth
            "ct": torch.tensor(data["ct_zyx"], dtype=torch.float32)  # Ground truth 3D CT volume (if saved)
        }

# Define the 3D U-Net model
class UNet3D_CT(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16, depth=4):
        super().__init__()
        self.inc = self.double_conv(in_ch, base)
        self.down1 = self.down_block(base, base * 2)
        self.down2 = self.down_block(base * 2, base * 4)
        self.down3 = self.down_block(base * 4, base * 8)
        self.up1 = self.up_block(base * 8, base * 4)
        self.up2 = self.up_block(base * 4, base * 2)
        self.up3 = self.up_block(base * 2, base)
        self.outc = nn.Conv3d(base, out_ch, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool3d(2),
            self.double_conv(in_ch, out_ch)
        )

    def up_block(self, in_ch, out_ch):
        return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.up2(x + x3)
        x = self.up3(x + x2)
        x = self.outc(x + x1)
        return x

# Training settings
device = "cuda" if torch.cuda.is_available() else "cpu"
npz_dir = Path(r"C:\Users\catal\XRAY_VIEW_SYNTHESIS\runs\runs_final\data\drr_pairs_fixed\npz")
batch_size = 1
num_epochs = 50
learning_rate = 1e-4

# Create DataLoader
dataset = DRRPairsDataset(npz_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
model = UNet3D_CT(in_ch=1, out_ch=1, base=16, depth=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        bp = batch["bp"].to(device)
        lat_gt = batch["lat"].to(device)

        # Forward pass: Predict the CT from back-projected volume
        ct_pred = model(bp)

        # Forward-project the predicted CT to LAT
        lat_pred = ct_pred  # You can modify this if you want to forward-project it to LAT

        # Compute loss (MSE loss between predicted and ground truth LAT)
        loss = torch.nn.functional.mse_loss(lat_pred, lat_gt)

        # Backpropagation and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    # Optionally save the model every epoch
    if (epoch + 1) % 5 == 0:  # Save every 5 epochs
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/unet3d_epoch_{epoch + 1}.pth")

# Save the final model after training
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/unet3d_final.pth")
print("Training complete. Final model saved.")
