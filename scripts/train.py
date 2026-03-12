import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from model import CLS_Classifier
import os

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.Resize(32),        
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# Le dossier data sera créé dans classifier_guidance_diffusion_model/data
dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# Configuration des classes pour MNIST
num_classes = 10
classes = [str(i) for i in range(10)] # ["0", "1", "2", ..., "9"]
# --- Model Initialization ---
classifier = CLS_Classifier(
    source_channel=1, 
    num_classes=10,
    unet_base_channel=128,
    num_norm_groups=32,
    head_dim=64,
).to(device)

# --- Préparation du stockage ---
checkpoint_dir = "../classifier_models" # Stockage en dehors de /scripts
os.makedirs(checkpoint_dir, exist_ok=True)

log_file = "train_loss.log"
if os.path.exists(log_file):
    os.remove(log_file)

# --- Configuration de l'optimiseur et du scheduler de bruit ---
opt = torch.optim.AdamW(classifier.parameters(), lr=3e-4, weight_decay=0.05)
T = 1000
# On utilise float64 pour la précision du cumulprod
alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

# --- Boucle d'entraînement ---
num_epochs = 480
print(f"Lancement de l'entraînement sur {device} pour {num_epochs} époques...")

for epoch_idx in range(num_epochs):
    epoch_loss = []
    classifier.train() 
    
    for data, label in loader:
        opt.zero_grad()

        x_0 = data.to(device)
        y = label.to(device)
        b = x_0.size(0)
        
        t = torch.randint(0, T, (b,)).to(device)
        eps = torch.randn_like(x_0).to(device)

        # Application du bruit (Reparameterization trick)
        s_alpha = sqrt_alpha_bars_t[t][:, None, None, None].float()
        s_one_minus = sqrt_one_minus_alpha_bars_t[t][:, None, None, None].float()
        x_t = s_alpha * x_0 + s_one_minus * eps

        # Forward
        logits = classifier(x_t, t)
        loss = F.cross_entropy(logits, y, reduction="mean")
        
        # Backward
        loss.backward()
        opt.step()

        epoch_loss.append(loss.item())

    # --- Fin de l'époque ---
    actual_epoch = epoch_idx + 1
    
    # Écriture logs
    with open(log_file, "a") as f:
        for l in epoch_loss:
            f.write(f"{l}\n")
    
    # Sauvegarde toutes les 10 époques
    if actual_epoch % 10 == 0:
        save_path = os.path.join(checkpoint_dir, f"classifier_epoch_{actual_epoch}.pt")
        torch.save(classifier.state_dict(), save_path)
        print(f"Époque {actual_epoch}/{num_epochs} complétée - Modèle sauvegardé.")
    elif actual_epoch == 1:
        print(f"Époque 1 terminée. L'entraînement se poursuit...")

print("Entraînement terminé avec succès.")