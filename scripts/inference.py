import os
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
from model_ddpm import UNet    
from model import CLS_Classifier as Classifier  

# Détection automatique du device (HAL a des GPUs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(unet, classifier, class_name, class_list, gradient_scale, epoch, T, alphas, alpha_bars, num_row=10, num_col=10):
    unet.eval()
    classifier.eval()

    alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
    sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    sigma_t = torch.sqrt(sigma_t_squared)

    class_id_list = [i for i, v in enumerate(class_list) if v == class_name]
    if len(class_id_list) == 0:
        raise Exception(f"La classe '{class_name}' n'existe pas dans la liste.")
    y = class_id_list[0]
    y_batch = (torch.tensor(y).to(device)).repeat(num_row * num_col)

    x = torch.randn(num_row * num_col, 1, 32, 32).to(device)

    print(f"Generation: Classe {class_name} | GS: {gradient_scale} | Epoch: {epoch}")
    
    # 4. Boucle de Reverse Diffusion
    for t in tqdm.tqdm(reversed(range(T)), total=T, leave=False):
        t_batch = (torch.tensor(t).to(device)).repeat(num_row * num_col)
        
        # Obtenir mu (prédiction du UNet)
        with torch.no_grad():
            epsilon = unet(x, t_batch)
        
        mu = (1.0 / torch.sqrt(alphas[t])) * (x - ((1.0 - alphas[t]) / torch.sqrt(1.0 - alpha_bars[t])) * epsilon)

        # Guidance : Calcul du gradient du classifieur (si GS > 0)
        grad = 0
        if gradient_scale > 0:
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t_batch)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y_batch.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0]

        # Ajout du bruit z (sauf à la dernière étape)
        z = torch.randn_like(x).to(device) if t > 0 else torch.zeros_like(x).to(device)
        
        # Mise à jour de x avec le gradient du classifieur
        x = mu + gradient_scale * sigma_t_squared[t] * grad + sigma_t[t] * z

    # 5. Post-traitement
    x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
    x = torch.clamp(x, min=0.0, max=1.0)

    # 6. Dessin et Sauvegarde de la grille 10x10
    fig, axes = plt.subplots(num_row, num_col, figsize=(12, 12))
    fig.suptitle(f"Class {class_name} | Guidance Scale: {gradient_scale}", fontsize=16)
    
    for i in range(num_row * num_col):
        image = x[i].cpu().numpy().squeeze() 
        row, col = i // num_col, i % num_col
        ax = axes[row, col]
        ax.set_axis_off()
        ax.imshow(image)

    # Gestion du dossier figures
    os.makedirs("../figures", exist_ok=True)
    mode = "guided" if gradient_scale > 0 else "uncond"
    filename = f"../figures/epoch_{epoch}_class_{class_name}_{mode}_gs_{gradient_scale}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close() 
    print(f" ✅ Figure sauvegardée: {filename}")

# --- BLOC D'EXÉCUTION ---
if __name__ == "__main__":
    # 1. Initialisation des modèles
    unet = UNet(source_channel=1, unet_base_channel=128, num_norm_groups=32).to(device)
    classifier = Classifier(source_channel=1, num_classes=10, unet_base_channel=128, num_norm_groups=32, head_dim=64).to(device)
    
    # 2. Chargement des poids (Modèles finaux)
    unet.load_state_dict(torch.load("../diffusion_model_checkpoints/ddpm_unet_epoch_500.pt", map_location=device))
    classifier.load_state_dict(torch.load("../classifier_models/classifier_epoch_480.pt", map_location=device))
    
    # 3. Paramètres de diffusion
    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    class_list = [str(i) for i in range(10)]

    # 4. GÉNÉRATION MASSIVE (Tests pour tous les chiffres)
    guidance_scales = [-1,0.0, 2.0, 6.0]
    
    for digit in class_list:
        for gs in guidance_scales:
            run_inference(
                unet, 
                classifier, 
                class_name=digit, 
                class_list=class_list, 
                gradient_scale=gs, 
                epoch=500, 
                T=T, 
                alphas=alphas, 
                alpha_bars=alpha_bars,
                num_row=10, 
                num_col=10
            )

    print("\n🎉 Toutes les générations sont terminées sur HAL !")