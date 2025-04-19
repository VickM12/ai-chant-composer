
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# --- Load dataset ---
class MidiTokenDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append((item['input'], item['target']))
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, target_token = self.data[idx]
        input_ids = self.tokenizer.encode(input_tokens)
        target_id = self.tokenizer.encode(target_token)[0]
        return torch.tensor(input_ids), torch.tensor(target_id)

# --- Tokenizer ---
class SimpleTokenizer:
    def __init__(self, tokens):
        self.token_to_id = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text):
        return [self.token_to_id[tok] for tok in text.split() if tok in self.token_to_id]

    def decode(self, ids):
        return " ".join([self.id_to_token[i] for i in ids])

# --- Build vocab ---
def build_vocab(dataset_path):
    vocab = set()
    with open(dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            vocab.update(item['input'].split())
            vocab.add(item['target'])
    return sorted(list(vocab))

# --- Model ---
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.fc(out[:, -1, :])
        return out

# --- Training ---
def train_model(dataset_path, epochs=5, batch_size=32, seq_len=10):
    vocab = build_vocab(dataset_path)
    tokenizer = SimpleTokenizer(vocab)
    dataset = MidiTokenDataset(dataset_path, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleLSTM(len(vocab), 128, 256)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model, tokenizer

# Example usage:
model, tokenizer = train_model('chant_training_dataset.jsonl')
