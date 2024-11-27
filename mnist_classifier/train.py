import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from mnist_classifier.model import MNISTModel
from datetime import datetime
from tqdm import tqdm
import os

def train(num_epochs=1):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset with augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=num_epochs,
                                            pct_start=0.1)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            acc = pred.eq(target).float().mean()
            
            # Update metrics
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            running_acc = 0.9 * running_acc + 0.1 * acc.item()
            
            pbar.set_postfix({
                'loss': f'{running_loss:.4f}',
                'acc': f'{running_acc:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Save model if accuracy improved
        if running_acc > best_acc:
            best_acc = running_acc
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            acc_str = f"{running_acc:.2f}".replace(".", "p")
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/mnist_model_{timestamp}_acc{acc_str}.pth')

if __name__ == '__main__':
    train(num_epochs=1) 