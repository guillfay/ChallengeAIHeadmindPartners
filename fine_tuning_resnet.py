#### Ce fichier génère des embedding pour les images de train et de test en utilisant un modèle pré-entrainé ResNet50 fine tuné à prédire les classes de vêtement




import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Paths
csv_file = "product_list.csv"
images_folder = "DAM_modified"

# Load labels
product_data = pd.read_csv(csv_file)
id_to_label = dict(zip(product_data['MMC'], product_data['Product_BusinessUnitDesc']))
unique_labels = list(set(product_data['Product_BusinessUnitDesc']))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple transforms for validation
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, images_folder, id_to_label, transform, label_to_idx):
        self.images_folder = images_folder
        self.id_to_label = id_to_label
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.image_files = [f for f in os.listdir(images_folder) if f.split('.')[0] in id_to_label]
        print(f"Total images: {len(self.image_files)}")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.images_folder, image_file)
        image_id = image_file.split('.')[0]
        label = self.id_to_label[image_id]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.label_to_idx[label]
        return image, label_idx

    def __len__(self):
        return len(self.image_files)

# Create datasets with different transforms
full_dataset = CustomDataset(images_folder, id_to_label, train_transform, label_to_idx)

# Split into train and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Override transform for validation dataset
val_dataset.dataset.transform = val_transform

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# Model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_classes = len(unique_labels)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Progressive unfreezing
def unfreeze_layer(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True

# Initially freeze all layers except fc
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# Training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), 100 * correct / total

# Training loop with progressive unfreezing
print("Starting training...")
best_val_acc = 0
patience = 10
no_improve = 0


def training(epochs_per_stage):
    global best_val_acc  # Déclare best_val_acc comme une variable globale
    global patience
    global no_improve
    for epoch in range(epochs_per_stage):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
    
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Epoch [{epoch + 1}/{epochs_per_stage}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print("Early stopping triggered")
            break


# Stage 1: Only fc layer
print("Stage 1: Training fc layer")
training(10)

# Stage 2: Unfreeze layer4
print("Stage 2: Training layer4")
unfreeze_layer(model, 'layer4')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Repeat training loop for layer4 (same as above)
training(6)

# # Stage 3: Unfreeze layer3
# print("Stage 3: Training layer3")
# unfreeze_layer(model, 'layer3')
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00005)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# # Repeat training loop for layer3
# training(5)

print("Training completed.")

# Load best model for embedding extraction
model.load_state_dict(torch.load('best_model.pth'))
embedding_model = nn.Sequential(*list(model.children())[:-1])
embedding_model.eval()

def extract_embeddings(image_folder, transform, embedding_model):
    embeddings = {}
    image_files = [file for file in os.listdir(image_folder) if file.endswith((".png", ".jpg", ".jpeg"))]
    total_files = len(image_files)
    print(f"Total images to process: {total_files}")
    
    with torch.no_grad():
        for i, file in enumerate(image_files):
            image_path = os.path.join(image_folder, file)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            embedding = embedding_model(input_tensor)
            embedding = embedding.squeeze().cpu().numpy()
            
            embeddings[os.path.splitext(file)[0]] = embedding
            if (i + 1) % 100 == 0:
                print(f"Progress: {(i+1)/total_files*100:.2f}% ({i+1}/{total_files})")
    
    return embeddings

# Extract embeddings using the best model
print("Extracting embeddings...")
train_folder = './DAM_modified'
test_folder = './test_image_modified'

embeddings_train = extract_embeddings(train_folder, val_transform, embedding_model)
embeddings_test = extract_embeddings(test_folder, val_transform, embedding_model)

np.save("embeddings_train_modified_3.npy", embeddings_train)
np.save("embeddings_test_modified_3.npy", embeddings_test)
print("Embeddings extracted and saved.")




