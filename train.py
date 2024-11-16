import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the user provides the preprocessed dataset folder
if len(sys.argv) != 2:
    print("Usage: python3 train_resnet18.py <processed_folder_name>")
    sys.exit(1)

# Load preprocessed dataset folder from user input
dataset_dir = sys.argv[1]

if not os.path.exists(dataset_dir):
    print(f"Error: The folder '{dataset_dir}' does not exist.")
    sys.exit(1)

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize ResNet18 model
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 8)  # Adjusting for 8 classes
model = model.to(device)

# Optimizer, learning rate scheduler, and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the model
model_save_path = os.path.join(os.path.dirname(dataset_dir), "resnet18_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# Evaluate the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
