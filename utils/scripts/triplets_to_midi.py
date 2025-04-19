from mido import MidiFile, MidiTrack, Message

def parse_triplet_line(line):
    try:
        parts = line.strip().split()
        note = int(parts[0].split("_")[1])
        velocity = int(parts[1].split("_")[1])
        duration = int(parts[2].split("_")[1])
        return note, velocity, duration
    except (IndexError, ValueError):
        return None

def triplets_to_midi(input_path, output_path="converted_output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    with open(input_path, "r") as f:
        for line in f:
            triplet = parse_triplet_line(line)
            if triplet:
                note, velocity, duration = triplet
                track.append(Message('note_on', note=note, velocity=velocity, time=0))
                track.append(Message('note_off', note=note, velocity=0, time=duration))

    mid.save(output_path)
    print(f"🎵 MIDI file saved as: {output_path}")

if __name__ == "__main__":
    triplets_to_midi("chant_corpus_triplets.txt")
