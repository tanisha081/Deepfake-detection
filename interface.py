import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os


# Load Model Class
class DeepFakeModel(torch.nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # Updated weights argument
        self.base_model.classifier = torch.nn.Sequential(torch.nn.Linear(1280, 1), torch.nn.Sigmoid())

    def forward(self, x):
        return self.base_model(x)


# Paths
MODEL_PATH = "models/deepfake_detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = DeepFakeModel().to(device)

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
    exit(1)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Preprocessing Function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(image_path):
    """Predict if an image is deepfake or real."""
    if not os.path.exists(image_path):
        return "❌ Error: Image not found."

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).squeeze().cpu().item()

    return "🟢 Deepfake" if output > 0.5 else "🟢 Real"


# Example
if __name__ == "__main__":
    test_image = "test_image.jpg"  # Make sure this file exists in the correct directory!

    if not os.path.exists(test_image):
        print(f"❌ Error: Image '{test_image}' not found. Place it in the correct folder.")
    else:
        print(f"Prediction: {predict(test_image)}")
