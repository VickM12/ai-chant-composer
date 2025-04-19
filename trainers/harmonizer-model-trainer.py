import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os

class HarmonyDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append((item["melody"], item["harmony"]))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][0], dtype=torch.float32)
        y = torch.tensor(self.data[idx][1], dtype=torch.float32)
        return x, y

class HarmonizerLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def train_model(
    dataset_path="chant_harmony_training_dataset.jsonl",
    model_path="harmonizer_model.pt",
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3
):
    dataset = HarmonyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HarmonizerLSTM()

    if os.path.exists(model_path):
        print("🔁 Checkpoint found. Loading...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("⚙️ No checkpoint found. Training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for melody_batch, harmony_batch in dataloader:
            optimizer.zero_grad()
            preds = model(melody_batch)
            loss = loss_fn(preds, harmony_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), model_path)
        print(f"💾 Checkpoint saved to: {model_path}")

# Run this line if executing directly
train_model("chant_harmony_training_dataset.jsonl")