import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "data", "faces")
MODEL_PATH = os.path.join(BASE_DIR, "models", "liveness_model.pth")


class LivenessDataset(Dataset):
    """Dataset for liveness detection: live faces vs fake (spoof) images."""
    
    def __init__(self, faces_dir, transform=None, num_fake_per_live=2):
        self.samples = []
        self.transform = transform
        self.num_fake_per_live = num_fake_per_live
        
        # Load live face images
        live_images = []
        for person_dir in Path(faces_dir).iterdir():
            if person_dir.is_dir():
                for img_path in person_dir.glob("*.jpg"):
                    live_images.append((str(img_path), 1))  # label 1 = live
        
        print(f"Loaded {len(live_images)} live face images")
        
        # Generate synthetic fake (spoof) samples
        fake_images = []
        for img_path, _ in live_images:
            # Load image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            # Generate multiple fake versions
            for i in range(self.num_fake_per_live):
                fake_img = self._generate_fake_sample(img)
                # Save temporarily or store in memory
                fake_images.append((img_path, 0, fake_img))  # label 0 = fake
        
        print(f"Generated {len(fake_images)} synthetic fake samples")
        
        # Mix live and fake
        self.samples = live_images + [(path, label, None) for path, label, _ in fake_images]
        self.fake_samples = {path: img for path, _, img in fake_images if img is not None}
    
    def _generate_fake_sample(self, img):
        """Generate a synthetic fake (spoof) sample from a live image."""
        fake_img = img.copy()
        method = np.random.randint(0, 4)
        
        if method == 0:
            # Blur
            fake_img = fake_img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(3, 8)))
        elif method == 1:
            # Reduce colors (compress) to simulate photo of photo
            fake_img = fake_img.quantize(colors=np.random.randint(64, 256))
            fake_img = fake_img.convert("RGB")
        elif method == 2:
            # Add noise
            arr = np.array(fake_img, dtype=np.float32)
            noise = np.random.normal(0, 25, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            fake_img = Image.fromarray(arr)
        else:
            # Combination of blur + noise
            fake_img = fake_img.filter(ImageFilter.GaussianBlur(radius=2))
            arr = np.array(fake_img, dtype=np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            fake_img = Image.fromarray(arr)
        
        return fake_img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if len(self.samples[idx]) == 2:
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
        else:
            img_path, label, fake_img = self.samples[idx]
            img = fake_img if fake_img is not None else Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


class SimpleLivenessNet(nn.Module):
    """Simple CNN for liveness detection."""
    
    def __init__(self):
        super(SimpleLivenessNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes: fake (0), live (1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_liveness_model(faces_dir, model_path, epochs=20, batch_size=8, lr=0.001):
    """Train a liveness detection model."""
    
    print("=" * 60)
    print("Training Liveness Detection Model")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LivenessDataset(faces_dir, transform=transform, num_fake_per_live=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {batch_size}, Batches per epoch: {len(dataloader)}")
    
    # Model, optimizer, loss
    model = SimpleLivenessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("\n✅ Training complete!")
    
    # Save model weights (state_dict) instead of full model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
    
    return model


if __name__ == "__main__":
    train_liveness_model(FACES_DIR, MODEL_PATH, epochs=20, batch_size=4, lr=0.001)
