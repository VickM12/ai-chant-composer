import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os

class HarmonySequenceDataset(Dataset):
    def __init__(self, path, sequence_length=4):
        self.sequence_length = sequence_length
        self.data = []
        with open(path, 'r') as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
        for i in range(len(lines) - sequence_length):
            melody_seq = [lines[j]["melody"] for j in range(i, i + sequence_length)]
            harmony_target = lines[i + sequence_length]["harmony"]
            self.data.append((melody_seq, harmony_target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        melody_seq, harmony_target = self.data[idx]
        x = torch.tensor(melody_seq, dtype=torch.float32)  # (sequence_length, 3)
        y = torch.tensor(harmony_target, dtype=torch.float32)  # (3,)
        return x, y

class HarmonizerBiLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=3, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):  # x: (batch, seq_len, 3)
        out, _ = self.bilstm(x)
        out = self.dropout(out[:, -1, :])  # last hidden state from both directions
        return self.fc(out)  # (batch, output_size)

def train_harmonizer(
    dataset_path="chant_harmony_training_dataset.jsonl",
    model_path="harmonizer_model_v2.pt",
    sequence_length=4,
    num_epochs=15,
    batch_size=32,
    learning_rate=1e-3
):
    dataset = HarmonySequenceDataset(dataset_path, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HarmonizerBiLSTM()

    if os.path.exists(model_path):
        print("🔁 Loading checkpoint from previous run...")
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for melody_seq, harmony_target in dataloader:
            optimizer.zero_grad()
            preds = model(melody_seq)
            loss = loss_fn(preds, harmony_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📚 Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved checkpoint to {model_path}")

if __name__ == "__main__":
    train_harmonizer()
