# Cnn Model.py
import os
import csv
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# -------- Dataset Class --------
class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['segment_filename'] + '.png')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label, self.data.iloc[idx]['segment_filename']

# -------- CNN Model --------
def get_model(num_classes=2):
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------- Weighted Loss --------
def calculate_class_weights(dataset):
    labels = [label for _, label, _ in dataset]
    class_sample_count = torch.tensor([(labels.count(t)) for t in torch.unique(torch.tensor(labels), sorted=True)])
    weight = 1. / class_sample_count.float()
    return weight

# -------- Training Function --------
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\n🌀 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, _ in tqdm(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"✅ Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

# -------- Run Pipeline --------
if __name__ == '__main__':
    csv_file = 'D:/Untliteled 17/spectograms/physionet_labels_multi_segment.csv'
    root_dir = 'D:/Untliteled 17/spectograms'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    full_dataset = SpectrogramDataset(csv_file, root_dir, transform=transform)

    # Train/Val/Test (70/15/15)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32),
        'test': DataLoader(test_dataset, batch_size=32)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = 'D:/Untliteled 17/Main Model/spectrogram_model.pth'
    model = get_model()

    if os.path.exists(model_path):
        print("🚀 Found saved model. Loading...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("🚀 No saved model found or training forced. Training from scratch...")

        class_weights = calculate_class_weights(train_dataset)
        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_model(model, dataloaders, criterion, optimizer, device)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to: {model_path}")

    # ✅ evaluation 
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for inputs, labels, filenames in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)

    test_acc = correct / total
    print(f"📊 Test Accuracy: {test_acc:.4f}")

    # ✅ save the prediction result in csv file 
    output_csv_path = 'D:/Untliteled 17/Main Model/test_predictions.csv'
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['segment_filename', 'True Label', 'Predicted Label'])
        writer.writerows(zip(all_filenames, all_labels, all_preds))

    print(f"📁 Predictions saved to: {output_csv_path}")
