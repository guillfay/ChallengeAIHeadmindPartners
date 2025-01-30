#### Ce fichier génère des embedding pour les images de test et les images DAM avec (et sans data augmentation) avec Clip



import os
import numpy as np
import torch
from PIL import Image
import clip
from torchvision import transforms

# Fonction pour extraire les embeddings sans data augmentation
def extract_embeddings_without_augmentation(folder_path, output_file, model, preprocess, device):
    image_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
    embeddings = {}

    with torch.no_grad():  # Pas besoin de calculer les gradients
        for file in image_files:
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path).convert("RGB")

            # Appliquer les transformations standard (sans augmentation)
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            # Extraire l'embedding
            embedding = model.encode_image(input_tensor).cpu().numpy().squeeze()

            # Sauvegarder l'embedding avec le nom du fichier (sans extension)
            embeddings[os.path.splitext(file)[0]] = embedding

    # Sauvegarder les embeddings dans un fichier
    np.save(output_file, embeddings)
    print(f"Embeddings sans data augmentation extraits et sauvegardés dans {output_file}.")

# Fonction pour extraire les embeddings avec data augmentation
def extract_embeddings_with_augmentation(folder_path, output_file, model, preprocess, device, num_augmentations=5):
    image_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
    embeddings = {}

    augmentation_transforms = transforms.Compose([
        transforms.ColorJitter(hue=0.2, saturation=0.3, brightness=0.3, contrast=0.3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():  # Pas besoin de calculer les gradients
        for file in image_files:
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path).convert("RGB")

            # Ajouter l'embedding de l'image originale
            original_tensor = preprocess(image).unsqueeze(0).to(device)
            original_embedding = model.encode_image(original_tensor).cpu().numpy().squeeze()
            embeddings[os.path.splitext(file)[0]] = original_embedding

            # Générer et ajouter les embeddings augmentés
            for i in range(num_augmentations):
                augmented_image = augmentation_transforms(image)
                augmented_tensor = augmented_image.unsqueeze(0).to(device)
                augmented_embedding = model.encode_image(augmented_tensor).cpu().numpy().squeeze()
                augmented_key = f"{os.path.splitext(file)[0]}_aug_{i+1}"
                embeddings[augmented_key] = augmented_embedding

    # Sauvegarder les embeddings dans un fichier
    np.save(output_file, embeddings)
    print(f"Embeddings avec {num_augmentations} augmentations par image extraits et sauvegardés dans {output_file}.")

# Charger le modèle CLIP et le preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # Utilise CLIP avec le modèle ViT-B/32
model.eval()  # Mettre le modèle en mode évaluation

# Extraire les embeddings sans augmentation
test_folder = "./test_image_modified"
test_embeddings_file = "embeddings_test_clip.npy"
extract_embeddings_without_augmentation(test_folder, test_embeddings_file, model, preprocess, device)

# Charger et afficher les clés des embeddings sans augmentation
embeddings_test = np.load(test_embeddings_file, allow_pickle=True).item()
print(f"Clés des embeddings de test : {list(embeddings_test.keys())[:10]}")

# Extraire les embeddings avec augmentation
dam_folder = "./DAM_modified"
dam_embeddings_file = "embeddings_dam_clip_combined.npy"
extract_embeddings_with_augmentation(dam_folder, dam_embeddings_file, model, preprocess, device, num_augmentations=5)

# Charger et afficher les clés des embeddings avec augmentation
embeddings_dam = np.load(dam_embeddings_file, allow_pickle=True).item()
print(f"Clés des embeddings DAM combinés : {list(embeddings_dam.keys())[:10]}")
