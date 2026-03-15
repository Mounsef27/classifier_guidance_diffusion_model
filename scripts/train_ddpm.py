import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from model_ddpm import UNet
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
unet = UNet(
    source_channel=1,
    unet_base_channel=128,
    num_norm_groups=32,
).to(device)

# --- Préparation du stockage ---
checkpoint_dir = "../diffusion_model_checkpoints" # Stockage en dehors de /scripts
os.makedirs(checkpoint_dir, exist_ok=True)

log_file ="train_diffusion.log"
if os.path.exists(log_file):
    os.remove(log_file)

# --- Configuration de l'optimiseur et du scheduler de bruit ---
opt = torch.optim.AdamW(unet.parameters(), lr=3e-4, weight_decay=0.05)
T = 1000
# On utilise float64 pour la précision du cumulprod
alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)
scheduler = torch.optim.lr_scheduler.LinearLR(
    opt,
    start_factor=1.0/5000,
    end_factor=1.0,
    total_iters=5000)
# loop
num_epochs = 500
method ='ddpm' # [ddpm ,ddim,sde]
if method=='ddpm':
    
    for epoch_idx in range(num_epochs):
        epoch_loss = []
        for batch_idx, (data, _) in enumerate(loader):
            unet.train()
            opt.zero_grad()

            # 2. Pick up x_0 (shape: [batch_size, 3, 32, 32])
            x_0 = data.to(device)

            # 3. Pick up random timestep, t .
            #    Instead of picking up t=1,2, ... ,T ,
            #    here we pick up t=0,1, ... ,T-1 .
            #   (i.e, t == 0 means diffused for 1 step)
            b = x_0.size(dim=0)
            t = torch.randint(T, (b,)).to(device)

            # 4. Generate the seed of noise, epsilon .
            #    We just pick up from 1D standard normal distribution with the same shape,
            #    because off-diagonal elements in covariance is all zero.
            eps = torch.randn_like(x_0).to(device)

            # 5. Compute x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
            #    (t == 0 means diffused for 1 step)
            x_t = sqrt_alpha_bars_t[t][:,None,None,None].float() * x_0 + sqrt_one_minus_alpha_bars_t[t][:,None,None,None].float() * eps

            # 6. Get loss and apply gradient (update)
            model_out = unet(x_t, t)
            loss = F.mse_loss(model_out, eps, reduction="mean")
            loss.backward()
            opt.step()
            scheduler.step()

            # log
            epoch_loss.append(loss.item())
            print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, loss), end="\r")

        # finalize epoch (save log and checkpoint)
        epoch_average_loss = sum(epoch_loss)/len(epoch_loss)
        print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, epoch_average_loss))
        with open(log_file, "a") as f:
            for l in epoch_loss:
                f.write("%s\n" %l)
        # 2. Sauvegarde du modèle toutes les 10 époques
        actual_epoch = epoch_idx + 1
        if actual_epoch % 10 == 0:
            save_path = os.path.join(checkpoint_dir, f"ddpm_unet_epoch_{actual_epoch}.pt")
            torch.save(unet.state_dict(), save_path)
            print(f"--- Checkpoint sauvegardé : {save_path} ---")
elif method == 'ddim':
    # TODO: Implémenter le sampling déterministe DDIM
    pass

elif method == 'sde':
    # TODO: Implémenter le score-based diffusion model
    pass

print("Entraînement terminé avec succès.")

