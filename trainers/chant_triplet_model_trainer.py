
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Dataset ---
class TripletSequenceDataset(Dataset):
    def __init__(self, path, seq_len):
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
        self.sequences = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        inp = self.sequences[idx]["input"]
        tgt = self.sequences[idx]["target"]

        notes = [triplet[0] for triplet in inp]
        velocities = [triplet[1] for triplet in inp]
        durations = [triplet[2] for triplet in inp]

        target_note = tgt[0]
        target_velocity = tgt[1]
        target_duration = tgt[2]

        return (
            torch.tensor(notes, dtype=torch.long),
            torch.tensor(velocities, dtype=torch.long),
            torch.tensor(durations, dtype=torch.long),
            torch.tensor(target_note, dtype=torch.long),
            torch.tensor(target_velocity, dtype=torch.long),
            torch.tensor(target_duration, dtype=torch.long),
        )

# --- Model ---
class StructuredLSTMModel(nn.Module):
    def __init__(self, note_dim=128, velocity_dim=128, duration_dim=1024, embed_size=32, hidden_size=128):
        super().__init__()
        self.note_embed = nn.Embedding(note_dim, embed_size)
        self.velocity_embed = nn.Embedding(velocity_dim, embed_size)
        self.duration_embed = nn.Embedding(duration_dim, embed_size)

        self.lstm = nn.LSTM(embed_size * 3, hidden_size, batch_first=True)
        self.fc_note = nn.Linear(hidden_size, note_dim)
        self.fc_velocity = nn.Linear(hidden_size, velocity_dim)
        self.fc_duration = nn.Linear(hidden_size, duration_dim)

    def forward(self, note_seq, velocity_seq, duration_seq):
        note_emb = self.note_embed(note_seq)
        vel_emb = self.velocity_embed(velocity_seq)
        dur_emb = self.duration_embed(duration_seq)

        x = torch.cat([note_emb, vel_emb, dur_emb], dim=-1)
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]

        return self.fc_note(last_output), self.fc_velocity(last_output), self.fc_duration(last_output)

# --- Training ---
def train_model(path, epochs=5, batch_size=32, seq_len=10):
    dataset = TripletSequenceDataset(path, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StructuredLSTMModel()
    checkpoint_path = "chant_triplet_model.pt"
    if os.path.exists(checkpoint_path):
        print("🔁 Loading existing model checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for n, v, d, tn, tv, td in loader:
            optimizer.zero_grad()
            pred_n, pred_v, pred_d = model(n, v, d)
            loss = loss_fn(pred_n, tn) + loss_fn(pred_v, tv) + loss_fn(pred_d, td)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        with open("epoch_log.txt", "a") as log:
            log.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}\n")

    torch.save(model.state_dict(), checkpoint_path)
    print("✅ Model saved as 'chant_triplet_model.pt'")

# --- Run ---
if __name__ == "__main__":
    train_model("utils/training-sets/chant_corpus_triplets_dataset.jsonl")
