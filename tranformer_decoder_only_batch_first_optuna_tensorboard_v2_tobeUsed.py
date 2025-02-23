import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import optuna
from torch.utils.tensorboard import SummaryWriter

# 創建數據集
class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=20):
        self.data = []
        self.vocab_size = vocab_size
        for _ in range(num_samples):
            seq = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]
            self.data.append(seq)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq

# 設定裝置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定義 Transformer Decoder-only 模型
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        return self.fc_out(x)

# 定義訓練和驗證函數
def train(model, loader, optimizer, criterion, writer, epoch, trial):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, model.fc_out.out_features), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        writer.add_scalar(f'Trial_{trial.number}/Train Loss', loss.item(), epoch * len(loader) + batch_idx)
    return total_loss / len(loader)

def validate(model, loader, criterion, writer, epoch, trial):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.fc_out.out_features), targets.view(-1))
            total_loss += loss.item()
    writer.add_scalar(f'Trial_{trial.number}/Val Loss', total_loss / len(loader), epoch)
    return total_loss / len(loader)

# 定義數據集和 DataLoader（移至 objective 外）
vocab_size = 20
max_seq_len = 10
batch_size = 16
train_dataset = SequenceDataset(800, max_seq_len, vocab_size)
val_dataset = SequenceDataset(200, max_seq_len, vocab_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 超參數調優函數
def objective(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    ff_dim = trial.suggest_categorical("ff_dim", [128, 256, 512, 1024])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    
    # Reinitialize model, optimizer, and loss function for each trial
    model = TransformerDecoderModel(vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'./runs/trial_{trial.number}')
    
    for epoch in range(5):
        train_loss = train(model, train_loader, optimizer, criterion, writer, epoch, trial)
        val_loss = validate(model, val_loader, criterion, writer, epoch, trial)
        trial.report(val_loss, epoch)
        print(f"Trial {trial.number}, Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if trial.should_prune():
            writer.close()
            raise optuna.exceptions.TrialPruned()
    
    writer.close()
    return val_loss

# 執行 Optuna 調參
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# 顯示最佳超參數
print("Best hyperparameters:", study.best_params)
