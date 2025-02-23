import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

# 創建數據集
class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=MAX_SEQ_LEN):
        self.data = []
        for _ in range(num_samples):
            seq = [random.randint(1, VOCAB_SIZE - 1) for _ in range(seq_len)]
            self.data.append(seq)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq

# 定義 Transformer Decoder-only 模型
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._get_SinusoidalPositionalEncoding(embed_dim, max_seq_len)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(DEVICE)
        x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        return self.fc_out(x)

    def _get_SinusoidalPositionalEncoding(self, d_model, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)
    
# 設定超參數
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
VOCAB_SIZE = 20  # 簡單的數字 token
MAX_SEQ_LEN = 10
BATCH_SIZE = 16
EPOCHS = 2
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = SequenceDataset(800)
val_dataset = SequenceDataset(200)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 初始化模型
model = TransformerDecoderModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# 訓練迴圈
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 測試生成
def generate_sequence(model, start_token, max_len=MAX_SEQ_LEN):
    model.eval()
    generated = [start_token]
    with torch.no_grad():
        for _ in range(max_len - 1):
            input_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            generated.append(next_token)
    return generated

# 測試生成序列
start_token = random.randint(1, VOCAB_SIZE - 1)
print("Generated Sequence:", generate_sequence(model, start_token))
