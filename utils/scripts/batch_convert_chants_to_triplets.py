
import os
from mido import MidiFile

def convert_folder_to_triplets(folder_path, output_path, min_velocity=40, max_velocity=80):
    triplets = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            try:
                filepath = os.path.join(folder_path, filename)
                midi = MidiFile(filepath)
                note_on_times = {}
                for track in midi.tracks:
                    current_time = 0
                    for msg in track:
                        current_time += msg.time
                        if msg.type == 'note_on' and msg.velocity > 0:
                            note_on_times[msg.note] = (current_time, msg.velocity)
                        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_on_times:
                            start_time, velocity = note_on_times.pop(msg.note)
                            duration = current_time - start_time
                            velocity = max(min_velocity, min(velocity, max_velocity))
                            triplets.append((msg.note, velocity, max(1, duration)))
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"⚠️ Skipped {filename}: {e}")

    with open(output_path, "w") as f:
        for note, velocity, duration in triplets:
            f.write(f"NOTE_{note} VELOCITY_{velocity} DURATION_{duration}\n")

    print(f"🎼 All triplets written to: {output_path}")

# Example usage
if __name__ == "__main__":
    convert_folder_to_triplets("utils/midi-files/gregorian-chants", "chant_corpus_triplets.txt")
