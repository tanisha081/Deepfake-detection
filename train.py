import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = "frames/"
REAL_PATH = os.path.join(DATA_PATH, "real")
FAKE_PATH = os.path.join(DATA_PATH, "fake")
MODEL_PATH = "models/deepfake_detector/epochs20.pth"
os.makedirs("models", exist_ok=True)

# Collect image paths and labels
image_paths = []
labels = []

for label, folder in enumerate([REAL_PATH, FAKE_PATH]):  # 0 for real, 1 for fake
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        if img_path.lower().endswith((".png", ".jpg", ".jpeg")):  # Ensure it's an image
            image_paths.append(img_path)
            labels.append(label)

# Train-test split
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2,
                                                                      random_state=42)


# Dataset Class
class DeepFakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataloaders
train_dataset = DeepFakeDataset(train_paths, train_labels, transform)
test_dataset = DeepFakeDataset(test_paths, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Model
class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.classifier = nn.Sequential(nn.Linear(1280, 1), nn.Sigmoid())

    def forward(self, x):
        return self.base_model(x)


# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeModel().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Progress Bar
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0  # Track loss per epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())  # Update progress bar with current loss

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Completed - Avg Loss: {avg_loss:.4f}")


# Save Model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
