import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# -------------------------
# 1. 建立完整的 Vocabulary
# -------------------------
VOCAB = [
    "[Start_of_Cell]", "[End_of_Cell]",
    "[Start_of_Description]", "[End_of_Description]", "[Padding_Description]",
    "[Start_of_Layout]", "[End_of_Layout]",
    "[Start_of_Rectangle]", "[End_of_Rectangle]", "[Padding_Rectangle]"
]

# 加入 9 個 layer token (M1-M5, V1-V4)
for layer in ["M1", "M2", "M3", "M4", "M5", "V1", "V2", "V3", "V4"]:
    VOCAB.append(f"LAYER_{layer}")

# 數字 token 與小數點 token
for i in range(10):
    VOCAB.append(f"digit_{i}")
VOCAB.append("dot")

# 建立 token2idx 與 idx2token
token2idx = {token: idx for idx, token in enumerate(VOCAB)}
idx2token = {idx: token for token, idx in token2idx.items()}

# -------------------------
# 2. 數值離散化函式
# -------------------------
def discretize_number(num, min_spacing=0.0005, max_range=10, precision=4):
    """
    將浮點數轉成固定格式的 token 序列：
    假設每個數字以 "digit_?" 與 "dot" 分隔，例: 0.1235 => ["digit_0", "dot", "digit_1", "digit_2", "digit_3", "digit_5"]
    """
    # 限制範圍
    num = max(min_spacing, min(num, max_range))
    # 格式化為固定小數點位數
    s = f"{num:.{precision}f}"
    tokens = []
    for ch in s:
        if ch == '.':
            tokens.append("dot")
        else:
            tokens.append(f"digit_{ch}")
    return tokens

# -------------------------
# 3. 描述文字 Tokenize 與 Padding
# -------------------------
def tokenize_description(text, max_len=50):
    """
    將描述文字轉成 token 序列，範例中簡單以字元分隔，實際上可換成其他 tokenize 方法。
    並固定長度 50，不足以 [Padding_Description] 補足。
    """
    # 這裡簡單以空白切分（或依需求使用更複雜 tokenize）
    tokens = text.split()
    tokens = tokens[:max_len]
    if len(tokens) < max_len:
        tokens += ["[Padding_Description]"] * (max_len - len(tokens))
    # 加上標記
    return ["[Start_of_Description]"] + tokens + ["[End_of_Description]"]

# -------------------------
# 4. 自定義資料集：讀取 CSV 並合併 rows（以 unitcell 為單位）
# -------------------------
class CellDataset(Dataset):
    def __init__(self, csv_file):
        super(CellDataset, self).__init__()
        # 假設 CSV 欄位包含: unitcell, description, layer, LLX, LLY, width, height
        self.df = pd.read_csv(csv_file)
        # 依 unitcell 分組
        self.grouped = self.df.groupby("unitcell")
        self.keys = list(self.grouped.groups.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        cell_name = self.keys[idx]
        group = self.grouped.get_group(cell_name)
        # 取得描述（假設每個 unitcell 只有一筆描述資料，或取第一筆）
        description = group["description"].iloc[0]
        desc_tokens = tokenize_description(description)
        
        # 產生版圖序列：包含 [Start_of_Layout] 與 [End_of_Layout]
        layout_tokens = ["[Start_of_Layout]"]
        # 假設每一筆代表一個 rectangle
        # 每個 rectangle 序列格式：
        # [Start_of_Rectangle]，接著 layer token，與四個座標數值離散化後 token 化，最後 [End_of_Rectangle]
        for _, row in group.iterrows():
            rect_tokens = ["[Start_of_Rectangle]"]
            # 取得 layer token (此處假設 CSV 中 layer 格式與 "M1", "V2" 等相符)
            rect_tokens.append(f"LAYER_{row['layer']}")
            # 處理四個數值：LLX, LLY, width, height
            for col in ["LLX", "LLY", "width", "height"]:
                num = float(row[col])
                rect_tokens.extend(discretize_number(num))
            rect_tokens.append("[End_of_Rectangle]")
            layout_tokens.extend(rect_tokens)
        # 若 rectangle 總數不足 100，則用 [Padding_Rectangle] 補足
        # 這裡先計算目前已加入的 rectangle數量
        num_rect = len(group)
        if num_rect < 100:
            for _ in range(100 - num_rect):
                layout_tokens.append("[Padding_Rectangle]")
        layout_tokens.append("[End_of_Layout]")
        
        # 組合完整 cell 序列
        full_tokens = ["[Start_of_Cell]"] + desc_tokens + layout_tokens + ["[End_of_Cell]"]
        # 將 token 轉換成 index 序列
        token_ids = [token2idx.get(token, 0) for token in full_tokens]  # 未知 token 預設 0
        return torch.tensor(token_ids, dtype=torch.long)

# -------------------------
# 5. 建立 Transformer Decoder-Only 模型
# -------------------------
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        # 注意：不加入 pos_embedding，因順序資訊由 token 本身隱含
    
    def forward(self, src, tgt_mask=None):
        # src: (seq_len, batch)
        embedded = self.embed(src) * math.sqrt(self.d_model)
        # 由於我們是 decoder-only 模型，這裡直接使用 transformer_decoder，令記憶庫為空
        # 輸入形狀轉為 (seq_len, batch, d_model)
        output = self.transformer_decoder(embedded, memory=None, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

# -------------------------
# 6. 訓練函式（Ray Tune 可呼叫）
# -------------------------
def train_cell(config, checkpoint_dir=None, csv_file="cells_data.csv"):
    # 設定 TensorBoard 寫入目錄（每個 trial 皆獨立）
    writer = SummaryWriter(log_dir=tune.get_trial_dir())
    
    # 建立資料集與 dataloader
    dataset = CellDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=token2idx["[Padding_Description]"]))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TransformerDecoderModel(vocab_size=len(VOCAB), d_model=config["d_model"], nhead=config["nhead"], num_layers=config["num_layers"])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # 如有 checkpoint，則讀取
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    criterion = nn.CrossEntropyLoss(ignore_index=token2idx["[Padding_Description]"])
    
    # 訓練 epochs
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            # batch: (batch, seq_len) -> 轉置成 (seq_len, batch)
            batch = batch.to(device).transpose(0, 1)
            optimizer.zero_grad()
            # 預測全部序列，但用前 n-1 預測第 2 ~ n 個 token
            logits = model(batch)
            # 將 logits 與 targets align: 將 logits 去掉最後一個時間步，target 去掉第一個 token
            logits = logits[:-1].reshape(-1, logits.shape[-1])
            targets = batch[1:].reshape(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        
        # 簡單驗證：取第一個 batch第一筆樣本，將幾何 token 解碼回座標（只針對 rectangle 部分）並印出檢查
        model.eval()
        with torch.no_grad():
            sample = batch[:, 0]
            # 取出 token 序列轉回 token list
            token_seq = [idx2token[idx.item()] for idx in sample]
            # (此處僅示範印出完整 token 序列，實際可寫解碼器將離散 token 轉回數值)
            if epoch % 2 == 0:
                print(f"Epoch {epoch} sample token sequence:")
                print(token_seq)
        
        # Ray Tune 專用：回報訓練 loss
        tune.report(loss=avg_loss)
        
        # 儲存 checkpoint
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
    
    writer.close()

# -------------------------
# 7. Ray Tune 主程式設定
# -------------------------
def main():
    config = {
        "d_model": tune.choice([128, 256]),
        "nhead": tune.choice([4, 8]),
        "num_layers": tune.choice([2, 4]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8]),
        "epochs": 5
    }
    
    scheduler = ASHAScheduler(
        max_t=config["epochs"],
        grace_period=1,
        reduction_factor=2
    )
    
    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])
    
    analysis = tune.run(
        train_cell,
        metric="loss",
        mode="min",
        config=config,
        num_samples=3,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="ray_cells_results",
        resources_per_trial={"cpu": 2, "gpu": 0},
        # 若 CSV 檔案不在當前路徑，可利用 Ray Tune 的 upload 功能上傳資料
    )
    
    best_trial = analysis.get_best_trial(metric="loss", mode="min", scope="all")
    print("最佳 trial 設定: {}".format(best_trial.config))
    print("最佳 trial 最終 loss: {}".format(best_trial.last_result["loss"]))

if __name__ == "__main__":
    main()
