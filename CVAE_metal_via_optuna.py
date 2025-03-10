import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import optuna

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
    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=128):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
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
    
    def forward(self, given_rect, generated_rect):
        x = torch.cat([given_rect, generated_rect], dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, given_rect)
        return recon_x, mu, logvar

# 訓練函數
def loss_function(recon_x, x, mu, logvar):
    mse_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + 0.0001 * kl_loss

# Optuna 目標函數
def objective(trial):
    latent_dim = trial.suggest_categorical("latent_dim", [2, 4, 8])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    # lr = trial.suggest_float("lr", 1e-4, 1e-3)
    lr = trial.suggest_categorical("lr", [1e-4, 1e-3])
    
    dataset = RectangleDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = CVAE(input_dim=4, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for given_rect, generated_rect in train_loader:
            given_rect, generated_rect = given_rect.to(device), generated_rect.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(given_rect, generated_rect)
            loss = loss_function(recon_x, generated_rect, mu, logvar)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for given_rect, generated_rect in val_loader:
                given_rect, generated_rect = given_rect.to(device), generated_rect.to(device)
                recon_x, mu, logvar = model(given_rect, generated_rect)
                loss = loss_function(recon_x, generated_rect, mu, logvar)
                val_loss += loss.item()
    
    return val_loss / len(val_loader)

# 執行 Optuna 超參數搜尋
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# 最佳超參數
print("Best hyperparameters:", study.best_params)
