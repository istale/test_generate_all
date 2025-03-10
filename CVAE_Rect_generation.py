import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 產生合成數據
class RectangleDataset(Dataset):
    def __init__(self, num_samples=100000):
        self.data = []
        for _ in range(num_samples):
            # 確保 w/h 或 h/w 至少大於 5
            while True:
                w_g = np.random.uniform(50, 300)  # given width
                h_g = np.random.uniform(10, 60)   # given height
                if max(w_g/h_g, h_g/w_g) > 5:
                    break

            x_g = np.random.uniform(0, 500)  # given rectangle x
            y_g = np.random.uniform(0, 500)  # given rectangle y

            # Generated rectangle 的 w = h = Given rectangle 的短邊
            short_side = min(w_g, h_g)
            w_r = h_r = short_side

            # Enclosure 值等於 Given rectangle 的短邊 * 2
            if w_g > h_g:
                x_r = x_g + w_g - short_side * 2
                y_r = y_g
            else:
                x_r = x_g
                y_r = y_g + h_g - short_side * 2

            # 進行 Normalization
            given_rect = [x_g / 500, y_g / 500, w_g / 300, h_g / 300]
            generated_rect = [x_r / 500, y_r / 500, w_r / 300, h_r / 300]

            self.data.append((given_rect, generated_rect))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        given_rect, generated_rect = self.data[idx]
        return torch.tensor(given_rect, dtype=torch.float32), torch.tensor(generated_rect, dtype=torch.float32)

# CVAE 模型
class CVAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, given_rect):
        z = torch.cat([z, given_rect], dim=1)
        
        return self.decoder(z)
    # def decode(self, z, given_rect):
    #     z = torch.cat([z, given_rect], dim=1)
    #     output = self.decoder(z)

    #     # 強制讓 w_r, h_r = 短邊
    #     short_side = torch.min(given_rect[:, 2], given_rect[:, 3])
    #     output[:, 2] = short_side
    #     output[:, 3] = short_side
    #     return output

    def forward(self, given_rect, generated_rect):
        x = torch.cat([given_rect, generated_rect], dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, given_rect)
        return recon_x, mu, logvar

# 訓練函數
# def loss_function(recon_x, x, mu, logvar):
#     mse_loss = nn.MSELoss()(recon_x, x)
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return mse_loss + 0.0001 * kl_loss

# 修正 Loss Function，強制 w_r, h_r = 短邊
def loss_function(recon_x, x, mu, logvar, given_rect):
    mse_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    short_side = torch.min(given_rect[:, 2], given_rect[:, 3])
    enclosure_loss = nn.MSELoss()(recon_x[:, 2], short_side) + nn.MSELoss()(recon_x[:, 3], short_side)

    return mse_loss + 0.0001 * kl_loss + 10.0 * enclosure_loss

# 訓練 CVAE
def train_cvae():
    dataset = RectangleDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 200
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for given_rect, generated_rect in train_loader:
            given_rect, generated_rect = given_rect.to(device), generated_rect.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(given_rect, generated_rect)
            # loss = loss_function(recon_x, generated_rect, mu, logvar)
            loss = loss_function(recon_x, generated_rect, mu, logvar, given_rect)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for given_rect, generated_rect in val_loader:
                given_rect, generated_rect = given_rect.to(device), generated_rect.to(device)
                recon_x, mu, logvar = model(given_rect, generated_rect)
                # loss = loss_function(recon_x, generated_rect, mu, logvar)
                loss = loss_function(recon_x, generated_rect, mu, logvar, given_rect)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.8f}, Val Loss: {val_loss / len(val_loader):.8f}")

    return model

# 訓練模型
model = train_cvae()

# 生成新 rectangle
def generate_rectangle(model, given_rect):
    model.eval()
    given_rect = torch.tensor(given_rect, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        z = torch.randn((1, model.latent_dim)).to(device)
        generated_rect = model.decode(z, given_rect)
    return generated_rect.cpu().numpy().flatten()

# 測試生成
sample_given_rect = [100, 200, 500, 50]
sample_generated_rect = generate_rectangle(model, sample_given_rect)
print("Given Rectangle:", sample_given_rect)
print("Generated Rectangle:", sample_generated_rect)

sample_given_rect = [100, 200, 500, 50]
for i in range(5):
    sample_generated_rect = generate_rectangle(model, sample_given_rect)
    print(f"Generated Rectangle {i+1}: {sample_generated_rect}")
