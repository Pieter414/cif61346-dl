import os, torch, numpy as np
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Config
N_SAMPLES = 1000
IMG_H = IMG_W = 5
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 5           # ubah ke 30/50 untuk training lebih lama
LATENT_DIM = 2
OUT_DIR = "./ae_random_5x5"
os.makedirs(OUT_DIR, exist_ok=True)

# Prepare random dataset (normalize to [-1,1])
X = np.random.rand(N_SAMPLES, 1, IMG_H, IMG_W).astype(np.float32)
X = (X - 0.5) / 0.5
dataset = TensorDataset(torch.from_numpy(X))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
class SmallConvAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(True)
        )
        self.fc_enc = nn.Linear(16 * IMG_H * IMG_W, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 16 * IMG_H * IMG_W)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, padding=1), nn.Tanh()
        )
    def encode(self, x):
        h = self.encoder_conv(x)
        return self.fc_enc(h.view(h.size(0), -1))
    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 16, IMG_H, IMG_W)
        return self.decoder_conv(h)
    def forward(self, x):
        return self.decode(self.encode(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallConvAE(latent_dim=LATENT_DIM).to(device)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Train (cepat)
for epoch in range(1, EPOCHS+1):
    model.train()
    run_loss = 0.0
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        opt.zero_grad()
        x_rec = model(x_batch)
        loss = loss_fn(x_rec, x_batch)
        loss.backward(); opt.step()
        run_loss += loss.item() * x_batch.size(0)
    print(f"Epoch {epoch}/{EPOCHS} Train MSE: {run_loss / N_SAMPLES:.6f}")

# Encode & reconstruct all samples
model.eval()
with torch.no_grad():
    X_t = torch.from_numpy(X).to(device)
    Z = model.encode(X_t).cpu().numpy()
    X_rec = model.decode(torch.from_numpy(Z).to(device)).cpu().numpy()

# Save model + images
torch.save(model.state_dict(), os.path.join(OUT_DIR, "ae_model.pth"))

# Latent scatter
plt.figure(figsize=(6,6))
plt.scatter(Z[:,0], Z[:,1], s=10)
plt.title("Latent space distribution (2D)")
plt.xlabel("z1"); plt.ylabel("z2")
plt.savefig(os.path.join(OUT_DIR, "latent_scatter.png")); plt.close()

# Reconstructions grid (stitched image)
n_examples = 12
orig = X[:n_examples,0]
rec  = X_rec[:n_examples,0]
row_orig = np.concatenate([orig[i] for i in range(n_examples)], axis=1)
row_rec  = np.concatenate([rec[i]  for i in range(n_examples)], axis=1)
stitched = np.concatenate([row_orig, row_rec], axis=0)
plt.figure(figsize=(12,4)); plt.imshow(stitched, cmap='gray', vmin=-1, vmax=1)
plt.title("Top: Originals | Bottom: Reconstructions"); plt.axis('off')
plt.savefig(os.path.join(OUT_DIR, "reconstructions.png")); plt.close()
