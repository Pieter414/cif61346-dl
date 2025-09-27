import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# CONFIG
BATCH_SIZE = 128
NUM_EPOCHS = 50
LR = 1e-3
OUT_DIR = "mnist_conv_autoencoder_images"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

################### SPECIFY THE DIRECTORIES AND TRANSFORMATIONS ###############################

trans = transforms.Compose([
  transforms.Resize(28),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5,), std=(0.5,))
])

################### CREATE THE DATASET OBJECT #################################################

trainfolder = MNIST(root='data', train=True, download=True, transform=trans)
testfolder  = MNIST(root='data', train=False, download=True, transform=trans)

trainloader = data.DataLoader(trainfolder, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
testloader  = data.DataLoader(testfolder, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

################### CREATE THE AUTOENCODER ####################################################

class MNISTAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=5),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

################ CREATE THE LOSS FUNCTION, OPTIMIZER, AND THE MODEL ##########################

ae = MNISTAE().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=LR)

################ TRAIN THE MODEL #############################################################

for epoch in range(1, NUM_EPOCHS+1):
    # TRAIN
    ae.train()
    train_loss_sum = 0.0
    for x, _ in tqdm(trainloader, desc=f"Epoch {epoch} train"):
        x = x.to(device)
        optimizer.zero_grad()
        x_pred = ae(x)
        loss = criterion(x_pred, x)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * x.size(0)  # sum over batch

    avg_train_loss = train_loss_sum / len(trainloader.dataset)

    # VALIDATION
    ae.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for x, _ in tqdm(testloader, desc=f"Epoch {epoch} valid"):
            x = x.to(device)
            x_pred = ae(x)
            loss = criterion(x_pred, x)
            test_loss_sum += loss.item() * x.size(0)

    avg_test_loss = test_loss_sum / len(testloader.dataset)

    # save one example (first batch)
    with torch.no_grad():
        x, _ = next(iter(testloader))
        x = x.to(device)
        x_pred = ae(x)
    # take the first sample in batch, channel 0
    pred_img = x_pred[0,0].cpu().numpy()  # shape (28,28)
    orig_img = x[0,0].cpu().numpy()

    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(2,1,1); ax1.imshow(pred_img, cmap='gray'); ax1.set_title('reconstruction'); ax1.axis('off')
    ax2 = fig.add_subplot(2,1,2); ax2.imshow(orig_img, cmap='gray'); ax2.set_title('original'); ax2.axis('off')
    fig.savefig(os.path.join(OUT_DIR, f"generated_image_epoch_{epoch}.png"))
    plt.close(fig)

    print(f"[{epoch}] Train Loss={avg_train_loss:.6f} Test Loss={avg_test_loss:.6f}")

################# SAVE THE IMAGE ##############################################################

torch.save(ae.state_dict(), "mnist_conv_autoencoder_weights.pth")