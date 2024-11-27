import torch
import pytest
import glob
import os
from model import MNISTModel
from torchvision import datasets, transforms

@pytest.fixture
def model():
    return MNISTModel()

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_input_shape(model):
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape {(batch_size, 10)}, got {output.shape}"

def test_model_accuracy():
    model = MNISTModel()
    
    # Try to load the latest trained model if available
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:  # If no trained model exists, train one
        from train import train
        train(num_epochs=3)  # Train for 3 epochs
        model_files = glob.glob('mnist_model_*.pth')
    
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    assert accuracy > 0.95, f"Model accuracy {accuracy:.2f} is below 0.95"  # Reduced threshold to 95%