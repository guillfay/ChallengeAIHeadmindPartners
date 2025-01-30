#### Ce fichier génère des embedding pour les images de test et les images DAM avec le modèle DINO v2.



import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Fonction pour extraire les embeddings sans data augmentation
def extract_embeddings_without_augmentation(folder_path, output_file, model, preprocess, device):
    image_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
    embeddings = {}

    with torch.no_grad():
        for file in image_files:
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            embedding = model(input_tensor).cpu().numpy().squeeze()
            embeddings[os.path.splitext(file)[0]] = embedding

    np.save(output_file, embeddings)
    print(f"Embeddings sans data augmentation extraits et sauvegardés dans {output_file}.")

# Fonction pour extraire les embeddings avec data augmentation
def extract_embeddings_with_augmentation(folder_path, output_file, model, preprocess, device, num_augmentations=5):
    image_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
    embeddings = {}

    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for file in image_files:
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path).convert("RGB")

            # Ajouter l'embedding de l'image originale
            original_tensor = preprocess(image).unsqueeze(0).to(device)
            original_embedding = model(original_tensor).cpu().numpy().squeeze()
            embeddings[os.path.splitext(file)[0]] = original_embedding

            # Générer et ajouter les embeddings augmentés
            for i in range(num_augmentations):
                augmented_image = augmentation_transforms(image)
                augmented_tensor = augmented_image.unsqueeze(0).to(device)
                augmented_embedding = model(augmented_tensor).cpu().numpy().squeeze()
                augmented_key = f"{os.path.splitext(file)[0]}_aug_{i + 1}"
                embeddings[augmented_key] = augmented_embedding

    np.save(output_file, embeddings)
    print(f"Embeddings avec {num_augmentations} augmentations par image extraits et sauvegardés dans {output_file}.")

# Charger le modèle DINO v2 depuis PyTorch Hub
device = "cuda" if torch.cuda.is_available() else "cpu"
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
dino.eval()

# Transformation des images pour DINO v2
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extraire les embeddings sans augmentation
test_folder = "./test_image_modified"
test_embeddings_file = "embeddings_test_dino.npy"
extract_embeddings_without_augmentation(test_folder, test_embeddings_file, dino, preprocess, device)

# Charger et afficher les clés des embeddings sans augmentation
embeddings_test = np.load(test_embeddings_file, allow_pickle=True).item()
print(f"Clés des embeddings de test : {list(embeddings_test.keys())[:10]}")

# Extraire les embeddings avec augmentation
dam_folder = "./DAM_modified"
dam_embeddings_file = "embeddings_dam_dinoV2_flipH_combined.npy"
extract_embeddings_with_augmentation(dam_folder, dam_embeddings_file, dino, preprocess, device, num_augmentations=5)

# Charger et afficher les clés des embeddings avec augmentation
embeddings_dam = np.load(dam_embeddings_file, allow_pickle=True).item()
print(f"Clés des embeddings DAM combinés : {list(embeddings_dam.keys())[:10]}")
