import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import time

# Configurações
dataset_path = "FairFace_Sample"
race_categories = ["black", "white"]
gender_categories = ["male", "female"]
img_size = 100
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

class FairFaceDataset(Dataset):
    def __init__(self, dataset_path, race_categories, gender_categories, img_size, transform=None):
        self.images, self.race_labels, self.gender_labels = [], [], []
        self.transform = transform
        for race_idx, race in enumerate(race_categories):
            for gender_idx, gender in enumerate(gender_categories):
                folder = os.path.join(dataset_path, f"{race}_{gender}")
                if not os.path.isdir(folder):
                    continue
                for filename in os.listdir(folder):
                    if filename.lower().endswith((".jpg", ".png")):
                        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (img_size, img_size))
                            self.images.append(img)
                            self.race_labels.append(race_idx)
                            self.gender_labels.append(gender_idx)
        self.race_labels = np.array(self.race_labels, dtype=np.float32)
        self.gender_labels = np.array(self.gender_labels, dtype=np.float32)

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img) if self.transform else torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        return img, torch.tensor(self.race_labels[idx]), torch.tensor(self.gender_labels[idx])

class MultiOutputCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size//8) * (img_size//8), 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc_race = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.fc_gender = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.fc_race(x), self.fc_gender(x)

# Dataset e DataLoader
full_dataset = FairFaceDataset(dataset_path, race_categories, gender_categories, img_size)
n_imgs = len(full_dataset)
epochs = n_imgs if n_imgs > 0 else 1
train_size = int(0.7 * n_imgs)
val_size = int(0.15 * n_imgs)
test_size = n_imgs - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = MultiOutputCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Treinamento
for epoch in range(1, epochs + 1):
    model.train()
    train_loss, total = 0, 0
    for imgs, race_labels, gender_labels in train_loader:
        imgs = imgs.to(device)
        race_labels = race_labels.to(device).unsqueeze(1)
        gender_labels = gender_labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        race_out, gender_out = model(imgs)
        loss = criterion(race_out, race_labels) + criterion(gender_out, gender_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    train_loss /= total

    model.eval()
    val_loss, total = 0, 0
    with torch.no_grad():
        for imgs, race_labels, gender_labels in val_loader:
            imgs = imgs.to(device)
            race_labels = race_labels.to(device).unsqueeze(1)
            gender_labels = gender_labels.to(device).unsqueeze(1)
            race_out, gender_out = model(imgs)
            loss = criterion(race_out, race_labels) + criterion(gender_out, gender_labels)
            val_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
    val_loss /= total
    print(f"Época {epoch} - Treino loss={train_loss:.4f} | Val loss={val_loss:.4f}")

# Webcam com moldura e timer
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Não foi possível acessar a webcam.")
else:
    start_time = time.time()
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    box = min(fw, fh) // 2
    top_left = ((fw - box) // 2, (fh - box) // 2)
    bottom_right = ((fw + box) // 2, (fh + box) // 2)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem da webcam.")
            cap.release()
            break
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
        cv2.putText(frame, "Enquadre seu rosto na moldura",
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        elapsed = time.time() - start_time
        remaining = max(0, 5 - int(elapsed))
        cv2.putText(frame, f"Captura em: {remaining}s",
                    (top_left[0], bottom_right[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Enquadre-se - Captura em 5 segundos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Captura cancelada pelo usuário.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        if elapsed > 5:
            break
    face_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    img = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    cap.release()
    cv2.destroyAllWindows()
    img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        race_pred, gender_pred = model(img_tensor)
        race_pred = race_pred.cpu().numpy()[0][0]
        gender_pred = gender_pred.cpu().numpy()[0][0]
    pred_race_class = race_categories[int(race_pred > 0.5)]
    pred_gender_class = gender_categories[int(gender_pred > 0.5)]
    confidence_race = race_pred if race_pred > 0.5 else 1 - race_pred
    confidence_gender = gender_pred if gender_pred > 0.5 else 1 - gender_pred
    print(f"\nA imagem capturada foi classificada como '{pred_race_class}' e '{pred_gender_class}' pela rede (confiança raça: {confidence_race*100:.2f}%, gênero: {confidence_gender*100:.2f}%)")