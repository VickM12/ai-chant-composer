from mido import MidiFile

# Path to your MIDI file
input_path = "02Track2_64kb.midi"
output_path = "midi_note_events.txt"

# Load the MIDI
midi = MidiFile(input_path)

# Open output file for writing
with open(output_path, "w") as f:
    for i, track in enumerate(midi.tracks):
        f.write(f'Track {i}: {track.name}\n')
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                f.write(f"{msg}\n")

print(f"Note events written to {output_path}")