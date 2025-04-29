import torch
from sklearn.metrics import accuracy_score, f1_score
from train import DeepFakeModel  # Import the model from train.py

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeModel().to(device)
model.load_state_dict(torch.load("models/deepfake_detector.pth"))
model.eval()

# Evaluate
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze().cpu().numpy()
        preds = (outputs > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
