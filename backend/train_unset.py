import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIG ---
DATA_DIR = "./IIASA_Dataset" # Point this to your downloaded IIASA data
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DATASET LOADER ---
class MiningDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = [x for x in os.listdir(os.path.join(root_dir, 'images')) if x.endswith('.png')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Load Image
        img_path = os.path.join(self.root_dir, 'images', img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask (Assuming mask has same filename)
        mask_path = os.path.join(self.root_dir, 'masks', img_name)
        mask = cv2.imread(mask_path, 0) # Load as grayscale
        mask = (mask > 0).astype('float32') # Convert to Binary 0.0 or 1.0

        # Preprocessing for ResNet
        image = image / 255.0 
        image = np.transpose(image, (2, 0, 1)).astype('float32')
        mask = np.expand_dims(mask, 0)

        return torch.tensor(image), torch.tensor(mask)

# --- TRAINING FUNCTION ---
def train():
    print(f"🚀 Training U-Net on {DEVICE}...")
    
    # 1. Create Model
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=3, 
        classes=1, 
        activation='sigmoid'
    )
    model.to(DEVICE)

    # 2. Setup Data
    dataset = MiningDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)

    # 3. Training Loop (Simplified)
    for epoch in range(10): # Run for 10-50 epochs
        total_loss = 0
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

    # 4. Save
    torch.save(model.state_dict(), "mineguard_unet_v1.pth")
    print("💾 Model saved as mineguard_unet_v1.pth")

if __name__ == "__main__":
    train()