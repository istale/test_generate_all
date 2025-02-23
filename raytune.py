import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# -------------------------
# 1. 定義 PyTorch 模型
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# 2. 定義訓練函式（Ray Tune 可呼叫）
# -------------------------
def train_mnist(config, checkpoint_dir=None):
    # 在 Ray Tune 執行環境下，每個 trial 都會有自己的日誌資料夾
    writer = SummaryWriter(log_dir=tune.get_trial_dir())
    
    # 建立模型與優化器
    model = SimpleNet(hidden_size=config["hidden_size"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # 資料準備
    # 這裡以 MNIST 為範例，你可替換成其他資料集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_full = datasets.MNIST(root="data", download=True, train=True, transform=transform)
    
    # 依據比例切分成訓練集與驗證集
    train_size = int(0.8 * len(mnist_full))
    val_size = len(mnist_full) - train_size
    train_dataset, val_dataset = random_split(mnist_full, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)
    
    # 如有 checkpoint，載入
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # 開始訓練
    for epoch in range(config["epochs"]):
        # 訓練階段
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        
        # 將結果記錄在 TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", accuracy, epoch)
        
        # Ray Tune 專用：回報當前 epoch 的指標給 Tune
        tune.report(loss=avg_val_loss, accuracy=accuracy)
        
        # 儲存 checkpoint
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), 
                path
            )
    
    writer.close()

# -------------------------
# 3. Ray Tune 執行設定
# -------------------------
def main():
    # 設定超參數搜尋空間
    config = {
        "hidden_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": 5
    }
    
    # 使用 ASHA 排程器作為篩選演算法
    scheduler = ASHAScheduler(
        max_t=config["epochs"],
        grace_period=1,
        reduction_factor=2
    )
    
    # 設定 CLIReporter 觀察想要的指標
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    
    # 執行 Ray Tune
    analysis = tune.run(
        train_mnist,
        metric="loss",               # 主要優化指標
        mode="min",                  # 因為是 loss，所以要找最小
        config=config,               # 參數搜尋空間
        num_samples=3,               # 每個組合重複試驗次數
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="ray_results",     # 儲存結果的資料夾
        resources_per_trial={"cpu": 2, "gpu": 0}  # 根據你的環境調整
    )
    
    # 取出最好的結果
    best_trial = analysis.get_best_trial(metric="loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    main()


# 使用 TensorBoard 進行視覺化

# 執行程式後，在終端機輸入（例如在 ray_results 目錄下）：

# tensorboard --logdir=ray_results
# 打開瀏覽器進入 http://localhost:6006/ (預設埠號 6006) 即可檢視多個試驗組合的曲線比較（損失、正確率等）。