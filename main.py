import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random

# --- Bloc 1 : Dataset et Transformation ---
class ImageDataset(Dataset):
    def __init__(self, image_folder, labels=None, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx] if self.labels else -1
        return image, label, image_path

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Bloc 2 : Chargement des données et des labels ---
# Paths
dam_folder = "./data/DAM/"
test_folder = "./data/test_image_headmind/"
product_list_path = "./data/product_list.csv"

# Charger les données
df = pd.read_csv(product_list_path)

# Filtrer les MMC et créer le mapping
classes = df["MMC"].unique()
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
index_to_class = {idx: cls for cls, idx in class_to_index.items()}
df["class_id"] = df["MMC"].map(class_to_index)

# Charger les datasets
labels = df["class_id"].tolist()
dam_dataset = ImageDataset(dam_folder, labels, transform=transform)
dam_loader = DataLoader(dam_dataset, batch_size=32, shuffle=True)

# --- Bloc 3 : Entraînement du modèle EfficientNet ---
# Charger un modèle pré-entraîné et ajuster la dernière couche
model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(classes))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Fonction d'entraînement
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Entraîner le modèle
train_model(model, dam_loader, criterion, optimizer, num_epochs=20)

# --- Bloc 4 : Extraction des Embeddings ---
model.eval()
dam_embeddings = []
dam_paths = []
with torch.no_grad():
    for images, _, paths in dam_loader:
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model(images).cpu().numpy()
        dam_embeddings.append(outputs)
        dam_paths.extend(paths)

dam_embeddings = np.vstack(dam_embeddings)

# --- Bloc 5 : Recherche de l'image la plus proche et sauvegarde ---
def save_comparison_subplots(test_image_path, model, dam_embeddings, dam_paths, output_dir="./output_subplots/"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    test_image = Image.open(test_image_path).convert("RGB")
    test_image_tensor = transform(test_image).unsqueeze(0).to(device)

    with torch.no_grad():
        test_embedding = model(test_image_tensor).cpu().numpy()

    # Similarité cosinus
    similarities = cosine_similarity(test_embedding, dam_embeddings)
    closest_idx = np.argmax(similarities)
    closest_image_path = dam_paths[closest_idx]

    # Création du subplot
    closest_image = Image.open(closest_image_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_image)
    axes[0].set_title("Test Image")
    axes[0].axis("off")
    axes[1].imshow(closest_image)
    axes[1].set_title("Closest DAM Image")
    axes[1].axis("off")

    # Sauvegarder le subplot
    output_path = os.path.join(output_dir, f"{os.path.basename(test_image_path)}_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison to {output_path}")

# --- Bloc 6 : Appliquer aux images de test ---
test_images = [os.path.join(test_folder, img) for img in os.listdir(test_folder) if img.lower().endswith((".jpeg", ".jpg", ".png"))]

for test_image_path in test_images:
    save_comparison_subplots(test_image_path, model, dam_embeddings, dam_paths)
