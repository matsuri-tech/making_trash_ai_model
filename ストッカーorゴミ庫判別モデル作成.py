import io
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from google.cloud import storage
from tqdm.notebook import tqdm

# -----------------------------------------
# 1. Cloud Storageから画像データを取得するDatasetの定義
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CloudStorageImageFolder(Dataset):
    def __init__(self, bucket_name, prefix, transform=None):
        """
        bucket_name: Cloud Storageのバケット名
        prefix: データセットのルートパス (例: "photos/ゴミ庫orストッカー判定/train")
        transform: 画像前処理
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        self.samples = []
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            parts = blob.name.split("/")
            if len(parts) < 5:
                continue
            label = parts[3]
            self.samples.append((blob.name, label))
        
        self.classes = sorted(list({label for _, label in self.samples}))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blob_name, label = self.samples[idx]
        blob = self.bucket.blob(blob_name)
        image_data = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx

# -----------------------------------------
# 2. Cloud Storage上の各データセット（train, val, test）の作成
# -----------------------------------------
bucket_name = "trash_check_ai"
train_prefix = "photos/ゴミ庫orストッカー判定/train"
val_prefix   = "photos/ゴミ庫orストッカー判定/val"
test_prefix  = "photos/ゴミ庫orストッカー判定/test"

train_dataset = CloudStorageImageFolder(bucket_name, train_prefix, transform=transform)
val_dataset   = CloudStorageImageFolder(bucket_name, val_prefix, transform=transform)
test_dataset  = CloudStorageImageFolder(bucket_name, test_prefix, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)
print("Class -> Index:", train_dataset.class_to_idx)
print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

# -----------------------------------------
# 3. モデルの作成と学習パイプラインの定義
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2クラス分類（環境に合わせて変更）
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    train_loss = running_loss / len(train_loader)
    
    # Validationループ
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_bar.set_postfix(loss=loss.item())
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# -----------------------------------------
# 4. テストデータで最終評価
# -----------------------------------------
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_loss = test_loss / len(test_loader)
test_acc = correct / total
print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# -----------------------------------------
# 5. 学習済みモデルの保存：Cloud Storage上の既存ファイルを上書き
# -----------------------------------------
# まずローカルに一時保存
local_model_path = "place.pth"
torch.save(model.state_dict(), local_model_path)

# Cloud Storageにアップロード（バケット名: trash_check_ai、Blob名: model/place.pth）
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob("model/place.pth")
blob.upload_from_filename(local_model_path)

print("Model saved to Cloud Storage at: trash_check_ai/model/place.pth")
