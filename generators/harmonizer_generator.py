
import torch
from advanced_harmonizer import HarmonizerBiLSTM

def generate_harmony(model_path, melody_sequence, sequence_length=4, output_tokens_path="generated_harmony_tokens.txt"):
    model = HarmonizerBiLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    harmony_triplets = []
    token_lines = []

    for i in range(len(melody_sequence) - sequence_length):
        window = melody_sequence[i:i+sequence_length]
        inp = torch.tensor([window], dtype=torch.float32)  # (1, sequence_length, 3)
        with torch.no_grad():
            pred = model(inp).squeeze().tolist()
            pred_note = int(round(pred[0]))
            pred_velocity = int(round(pred[1]))
            pred_duration = int(round(pred[2]))
            harmony_triplets.append([pred_note, pred_velocity, pred_duration])
            token_lines.append(f"Harmony: [{pred_note}, {pred_velocity}, {pred_duration}] from Melody: {window[-1]}")

    with open(output_tokens_path, "w") as f:
        f.write("\\n".join(token_lines))
    
    print(f"📝 Harmony tokens saved to {output_tokens_path}")
    return harmony_triplets