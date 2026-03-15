import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import sys

# Ajout du dossier courant au path pour l'import de model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import CLS_Classifier

def test_on_mnist():
    # 1. Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "classifier_models/classifier_epoch_480.pt"
    
    # 2. Initialisation du modèle avec TES paramètres exacts
    model = CLS_Classifier(
        source_channel=1, 
        num_classes=10,
        unet_base_channel=128,
        num_norm_groups=32,
        head_dim=64,
    ).to(device)
    
    # Chargement des poids
    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier {model_path} est introuvable !")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Modèle chargé depuis {model_path}")

    # 3. Préparation des données (Resize à 32 comme à l'entraînement !)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Chargement d'une image de test
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    image, label = next(iter(test_loader))
    image = image.to(device)
    T = 1000
    b = image.size(0)

    # 4. Inférence
    # Note : Le classifieur a besoin de 't'. Pour un test sur image propre, on peut mettre t=0.
    t = torch.randint(0, T, (b,)).to(device)
    
    with torch.no_grad():
        logits = model(image, t)
        prediction = logits.argmax(dim=1, keepdim=True)
        probs = F.softmax(logits, dim=1)

    # 5. Affichage et sauvegarde
    plt.figure(figsize=(6, 6))
    # On repasse en 28x28 ou on garde 32x32 pour l'affichage
    plt.imshow(image.cpu().squeeze())
    
    title = f"Vrai: {label.item()} | Pred: {prediction.item()} ({probs.max():.2%}| niveau de bruit : {t.item()})"
    plt.title(title)
    plt.axis('off')
    
    # Sauvegarde dans le dossier figures déjà existant
    save_path = f"figures/validation_mnist_{label.item()}.png"
    plt.savefig(save_path)
    plt.close()
    
    print(f"--- RÉSULTAT ---")
    print(f"Chiffre réel : {label.item()}")
    print(f"Prédiction   : {prediction.item()}")
    print(f"Confiance    : {probs.max():.2%}")
    print(f"Image sauvegardée dans : {save_path}")

if __name__ == "__main__":
    test_on_mnist()