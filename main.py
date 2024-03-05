import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from carbontracker.tracker import CarbonTracker
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from fastonn import SelfONN2d



class DenoisingDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir):
        
        self.root_dir = Path(root_dir)
        self.noisy_dir = self.root_dir / 'speckle_M1'
        self.non_noisy_dir = self.root_dir / 'clean'
        self.image_files = list(self.noisy_dir.glob('*.png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        noisy_img_path = self.image_files[idx]
        non_noisy_img_path = self.non_noisy_dir / noisy_img_path.name
        noisy_image = read_image(str(noisy_img_path)).float() / 255.0
        non_noisy_image = read_image(str(non_noisy_img_path)).float() / 255.0
        
        return noisy_image, non_noisy_image

class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17) -> None:
        super(DnCNN, self).__init__()

        bias = True

        # Head
        head = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )
        
        # Body
        body = []
        for _ in range(nb-2):
            body.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=bias))
            body.append(nn.BatchNorm2d(nc, momentum=0.9, eps=1e-04, affine=True))
            body.append(nn.ReLU(inplace=True))

        # Tail
        tail = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        # Combine head, body, and tail into a single model
        self.model = nn.Sequential(
            head,
            *body,
            tail
        )
        
    def forward(self, x):
        n = self.model(x)
        return x - n

class ShallowONN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        super(ShallowONN, self).__init__()
        self.onn1 = SelfONN2d(in_channels, 12, kernel_size, q=6, padding = 1)  # First operational layer
        self.onn2 = SelfONN2d(12, out_channels, kernel_size, q=6, padding = 1)  # Second operational layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = torch.relu(self.onn1(x))
        x = self.onn2(x)
        return x



def train_model(model, train_loader, criterion, optimizer, epochs=100):
    
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    log_dir = f"Logging"
    tracker = CarbonTracker(epochs=epochs, monitor_epochs=-1, log_dir=log_dir )

    for epoch in range(epochs):
        
        tracker.epoch_start()
        
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        
        tracker.epoch_end()
    
    tracker.stop()
        
    
    
        
def test_model(model, test_loader, criterion):
    
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():  # Inference mode, gradients not needed
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")



# Calculate PSNR
def calculate_psnr(target, output, max_pixel=1.0):
    
    mse = nn.functional.mse_loss(output, target)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def show_images(noisy_img, original_img, model, device):
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Load images
    noisy_image_tensor = read_image(noisy_img).float() / 255.0
    original_image_tensor = read_image(original_img).float() / 255.0
    
    # Add batch dimension
    noisy_image_tensor = noisy_image_tensor.unsqueeze(0).to(device)
    original_image_tensor = original_image_tensor.unsqueeze(0).to(device)
    
    # Denoise the image
    with torch.no_grad():
        denoised_image_tensor = model(noisy_image_tensor)
    
    # Calculate PSNR
    psnr = calculate_psnr(original_image_tensor, denoised_image_tensor)
    
    # Convert tensors to PIL images for display
    noisy_image_pil = to_pil_image(noisy_image_tensor.squeeze().cpu())
    denoised_image_pil = to_pil_image(denoised_image_tensor.squeeze().cpu())
    original_image_pil = to_pil_image(original_image_tensor.squeeze().cpu())
    
     # Display the images
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_pil, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image_pil, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image_pil, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')
    
    plt.show()
    
    # Print PSNR
    print(f"PSNR between original and denoised: {psnr} dB")



def main():
    
    root_dir = f"imgs/dataset"
    
    dataset = DenoisingDataset(root_dir)
    kf = KFold(n_splits=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 100
    
    

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)
        
        model = DnCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_model(model, train_loader, criterion, optimizer, epochs)
        test_model(model, test_loader, criterion)
    
        # calculate average PSNR over the test set
        model.eval()
        avg_psnr = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                avg_psnr += calculate_psnr(targets, outputs).item()
                
        avg_psnr /= len(test_loader)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        
        
    # show image
    noisy_path = f"imgs/dataset/test_images/speckle_noisy_M1_17124.png"
    original_path = f"imgs/dataset/test_images/original_17124.png"
    show_images(noisy_path, original_path, model, device)
        

if __name__ == "__main__":
    main()
