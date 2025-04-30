# train_pytorch.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 定義 MLP
class BreastCancerMLP(nn.Module):
    def __init__(self, input_size=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# 載入資料
data = load_breast_cancer()
X, y = data.data, data.target

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 資料轉成 DataLoader
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# 建立模型
model = BreastCancerMLP()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練
for epoch in range(20):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

# 儲存模型與標準化器
torch.save(model.state_dict(), "pytorch_model.pt")
joblib.dump(scaler, "pytorch_scaler.pkl")
print("✅ 儲存 pytorch_model.pt 和 pytorch_scaler.pkl 完成")

