
import torch
import json
from harmonizer_model import HarmonizerModel
from harmonizer_vocab import HarmonizerTokenizer
from chant_triplet_model_generator_diatonic import generate_melody_triplets  # Assumed melody source

def load_model(model_path="harmonizer_model.pt", vocab_path="harmonizer_vocab.json"):
    with open(vocab_path, 'r') as f:
        token_to_id = json.load(f)
    tokenizer = HarmonizerTokenizer(token_to_id)
    model = HarmonizerModel(len(token_to_id))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, tokenizer

def generate_harmony(model, tokenizer, melody_triplets):
    model.eval()
    harmony_triplets = []

    for triplet in melody_triplets:
        input_ids = tokenizer.encode_triplet(triplet)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            output = model(input_tensor)
        pred_ids = output.argmax(dim=-1).squeeze().tolist()
        harmony_triplet = tokenizer.decode_triplet(pred_ids)
        harmony_triplets.append(harmony_triplet)

    return harmony_triplets

def save_triplets_to_file(triplets, filename="generated_harmony_triplets.txt"):
    with open(filename, "w") as f:
        for note, velocity, duration in triplets:
            f.write(f"NOTE={note}, VELOCITY={velocity}, DURATION={duration}\n")
    print(f"✅ Harmony tokens saved to {filename}")

if __name__ == "__main__":
    model, tokenizer = load_model()
    melody_triplets = generate_melody_triplets()  # Replace or inject your own input
    harmony_triplets = generate_harmony(model, tokenizer, melody_triplets)
    save_triplets_to_file(harmony_triplets)
