
import torch
import torch.nn.functional as F
from mido import MidiFile, MidiTrack, Message
from random import randint
from chant_triplet_model_trainer import StructuredLSTMModel

def load_model(path='chant_triplet_model.pt'):
    model = StructuredLSTMModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def is_diatonic(note):
    return note % 12 in {0, 2, 4, 5, 7, 9, 11}  # C major/A minor scale

def generate_sequence(model, seed_triplets, length=120, temperature=0.9, output_token_file="generated_tokens_diatonic.txt", finalis=53):
    result = seed_triplets[:]
    tokens_written = []
    phrase_counter = 0

    with open(output_token_file, "w") as f:
        while len(tokens_written) < length:
            phrase_len = randint(8, 12)
            phrase_counter += 1

            for _ in range(phrase_len):
                note_seq = torch.tensor([[t[0] for t in result[-10:]]], dtype=torch.long)
                vel_seq = torch.tensor([[t[1] for t in result[-10:]]], dtype=torch.long)
                dur_seq = torch.tensor([[t[2] for t in result[-10:]]], dtype=torch.long)

                with torch.no_grad():
                    note_logits, vel_logits, dur_logits = model(note_seq, vel_seq, dur_seq)

                note_probs = F.softmax(note_logits[0] / temperature, dim=-1)
                vel_probs = F.softmax(vel_logits[0] / temperature, dim=-1)
                dur_probs = F.softmax(dur_logits[0] / temperature, dim=-1)

                note = torch.multinomial(note_probs, 1).item()
                velocity = torch.multinomial(vel_probs, 1).item()
                duration = torch.multinomial(dur_probs, 1).item()

                note = max(45, min(75, note))
                if not is_diatonic(note):
                    continue

                if abs(note - finalis) > 12:
                    note = finalis + (note - finalis) // 2

                velocity = max(50, min(80, velocity + randint(-5, 5)))
                duration = max(40, min(240, duration))

                triplet = [note, velocity, duration]
                if triplet in result[-3:]:
                    continue

                result.append(triplet)
                f.write(f"NOTE={note}, VELOCITY={velocity}, DURATION={duration}\n")
                tokens_written.append(triplet)

            if phrase_counter % 4 == 0:
                cadence = [finalis, 70, 240]
                result.append(cadence)
                f.write(f"NOTE={cadence[0]}, VELOCITY={cadence[1]}, DURATION={cadence[2]}  # CADENCE\n")
                tokens_written.append(cadence)

            result.append([0, 0, 240])
            f.write("NOTE=0, VELOCITY=0, DURATION=240  # REST\n")
            tokens_written.append([0, 0, 240])

        result.append([finalis, 64, 480])
        f.write(f"NOTE={finalis}, VELOCITY=64, DURATION=480  # FINAL TONE\n")
        result.append([0, 0, 480])
        f.write("NOTE=0, VELOCITY=0, DURATION=480  # END REST\n")

    return tokens_written

def save_as_midi(triplets, path="chant_triplet_output_diatonic.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note, velocity, duration in triplets:
        if note == 0 and velocity == 0:
            track.append(Message("note_off", note=60, velocity=0, time=duration))
        else:
            try:
                track.append(Message("note_on", note=note, velocity=velocity, time=0))
                track.append(Message("note_off", note=note, velocity=0, time=duration))
            except:
                continue

    mid.save(path)
    print(f"✅ MIDI saved as {path}")

if __name__ == "__main__":
    seed = [[60, 64, 80]] * 10
    model = load_model()
    generated = generate_sequence(model, seed, length=120)
    save_as_midi(generated)
