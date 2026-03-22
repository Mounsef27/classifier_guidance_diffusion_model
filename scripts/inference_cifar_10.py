import os
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model_ddpm import UNet    
from model import CLS_Classifier as Classifier  

# Détection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(unet, classifier, class_id, class_name, gradient_scale, epoch, T, alphas, alpha_bars, num_row=8, num_col=8):
    unet.eval()
    classifier.eval()

    # Paramètres de diffusion
    alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
    sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    sigma_t = torch.sqrt(sigma_t_squared)

    # Préparation du batch
    y_batch = torch.full((num_row * num_col,), class_id, dtype=torch.long).to(device)
    # CIFAR-10 = 3 canaux (RGB)
    x = torch.randn(num_row * num_col, 3, 32, 32).to(device)

    print(f">>> Génération : {class_name} (ID:{class_id}) | GS: {gradient_scale} | Epoch DDPM: {epoch}")
    
    for t in tqdm.tqdm(reversed(range(T)), total=T, leave=False):
        t_batch = torch.full((num_row * num_col,), t, dtype=torch.long).to(device)
        
        # 1. Prédiction du bruit par le UNet
        with torch.no_grad():
            epsilon = unet(x, t_batch)
        
        # 2. Calcul de mu (moyenne du reverse process)
        coeff1 = 1.0 / torch.sqrt(alphas[t])
        coeff2 = (1.0 - alphas[t]) / torch.sqrt(1.0 - alpha_bars[t])
        mu = coeff1 * (x - coeff2 * epsilon)

        # 3. Guidance du Classifieur
        if gradient_scale > 0:
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t_batch)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y_batch]
            grad = torch.autograd.grad(selected.sum(), x_in)[0]
            
            # Mise à jour de mu avec le gradient
            mu = mu + gradient_scale * sigma_t_squared[t] * grad

        # 4. Ajout du bruit z (Langevin dynamics)
        z = torch.randn_like(x).to(device) if t > 0 else 0
        x = mu + sigma_t[t] * z

    # 5. Post-traitement (Passage de [-1, 1] à [0, 1])
    x = (x.clamp(-1, 1) + 1) / 2
    x = x.permute(0, 2, 3, 1).cpu().numpy()

    # 6. Affichage et Sauvegarde
    fig, axes = plt.subplots(num_row, num_col, figsize=(12, 12))
    fig.suptitle(f"CIFAR-10: {class_name} | Guidance Scale: {gradient_scale}", fontsize=16)
    
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(x[i])
        ax.axis('off')

    os.makedirs("../figures", exist_ok=True)
    mode = "guided" if gradient_scale > 0 else "uncond"
    filename = f"../figures/cifar10_ep{epoch}_{class_name}_{mode}_gs{gradient_scale}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close() 
    print(f" ✅ Image enregistrée : {filename}")

if __name__ == "__main__":
    # Noms des classes CIFAR-10
    cifar10_classes = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

    unet = UNet(source_channel=3, unet_base_channel=128, num_norm_groups=32).to(device)
    classifier = Classifier(source_channel=3, num_classes=10, unet_base_channel=128, num_norm_groups=32, head_dim=64).to(device)
    
    unet.load_state_dict(torch.load("../diffusion_model_checkpoints_cifar_10/ddpm_unet_epoch_500.pt", map_location=device))
    classifier.load_state_dict(torch.load("../cifar_10_models/classifier_epoch_480.pt", map_location=device))
    
    T = 1000
    alphas = torch.linspace(0.9999, 0.98, T).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)

    target_classes = [0, 6 , 1, 3, 8] 
    scales = [0, 3, 10] # 0 = Sans guidance, 3-10 = Guidé
    
    for idx in target_classes:
        for gs in scales:
            run_inference(unet, classifier, idx, cifar10_classes[idx], gs, 330, T, alphas, alpha_bars)

    print("\n[Terminé] Toutes les images sont dans le dossier /figures !")