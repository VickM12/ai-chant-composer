
from mido import MidiFile

def convert_midi_to_triplets(input_path, output_path, min_velocity=40, max_velocity=80, default_velocity=64):
    midi = MidiFile(input_path)
    note_on_times = {}
    triplets = []

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
                triplets.append((msg.note, velocity, duration))

    with open(output_path, "w") as f:
        for note, velocity, duration in triplets:
            f.write(f"NOTE_{note} VELOCITY_{velocity} DURATION_{duration}\n")

    print(f"✅ Triplets written to: {output_path}")

# Example usage
if __name__ == "__main__":
    convert_midi_to_triplets("utils/midi-files/gregorian_chants/a-exsultate.mid", "chant_triplets.txt")
