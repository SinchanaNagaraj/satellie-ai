!pip install kagglehub torch torchvision matplotlib

import kagglehub, os

# Download dataset
path = kagglehub.dataset_download("mahmoudreda55/satellite-image-classification")

print("Dataset downloaded to:", path)
print("Folders inside:", os.listdir(path))
print("Class folders:", os.listdir(os.path.join(path, "data")))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=os.path.join(path, "data"), transform=transform)

# Split into train/val (80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)

print("Classes:", dataset.classes)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EfficientNet-B3 for stronger performance
model = models.efficientnet_b3(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=25):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

def evaluate_model(model, val_loader, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, val_loss / len(val_loader)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=5)

def calculate_accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Calculate train and validation accuracy
train_acc = calculate_accuracy(model, train_loader)
val_acc   = calculate_accuracy(model, val_loader)

print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

from PIL import Image
import torch.nn.functional as F

def predict_image(image_path, model, transform, classes):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    predicted_class = classes[pred.item()]
    confidence = probs[0][pred.item()].item()

    return predicted_class, confidence

import glob

# Pick one image from the "water" folder
test_image_path = glob.glob(os.path.join(path, "data", "water", "*.jpg"))[0]
print("Testing with:", test_image_path)

label, conf = predict_image(test_image_path, model, transform, dataset.classes)
print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")

import matplotlib.pyplot as plt
img = Image.open(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {label} ({conf*100:.2f}%)")
plt.axis("off")
plt.show()

files = glob.glob(os.path.join(path, "data", "green", "*.jpg"))
if len(files) == 0:
    print("No images found in 'green' folder. Available folders:", os.listdir(os.path.join(path, "data")))
else:
    test_image_path = files[0]
    print("Testing with:", test_image_path)
    label, conf = predict_image(test_image_path, model, transform, dataset.classes)
    print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")

print(os.listdir(os.path.join(path, "data")))

import glob

# Get one image path from the "green_area" (forest/vegetation) class folder
test_image_path = glob.glob(os.path.join(path, "data", "green_area", "*.jpg"))[0]
print("Testing with:", test_image_path)

label, conf = predict_image(test_image_path, model, transform, dataset.classes)
print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")

import matplotlib.pyplot as plt
img = Image.open(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {label} ({conf*100:.2f}%)")
plt.axis("off")
plt.show()

import glob
import matplotlib.pyplot as plt
from PIL import Image

# Pick one desert image from your dataset
test_image_path = glob.glob(os.path.join(path, "data", "desert", "*.jpg"))[0]
print("Testing with:", test_image_path)

# Run prediction
label, conf = predict_image(test_image_path, model, transform, dataset.classes)
print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")

# Display the image with prediction
img = Image.open(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {label} ({conf*100:.2f}%)")
plt.axis("off")
plt.show()

from google.colab import files
uploaded = files.upload()

# Replace with the filename you uploaded
test_image_path = "sahara.png"

label, conf = predict_image(test_image_path, model, transform, dataset.classes)
print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")

import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {label} ({conf*100:.2f}%)")
plt.axis("off")
plt.show()

from google.colab import files
uploaded = files.upload()



# Replace with the filename you uploaded
test_image_path = "mangrove.png"

label, conf = predict_image(test_image_path, model, transform, dataset.classes)
print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")



from google.colab import files
uploaded = files.upload()

# Replace with the filename you uploaded
test_image_path = "myphoto.jpg"

label, conf = predict_image(test_image_path, model, transform, dataset.classes)
print(f"Predicted Class: {label} | Confidence: {conf*100:.2f}%")

import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {label} ({conf*100:.2f}%)")
plt.axis("off")
plt.show()