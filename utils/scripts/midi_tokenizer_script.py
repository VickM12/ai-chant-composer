
from mido import MidiFile

# Path to your MIDI file
input_path = "02Track2_64kb.midi"
output_path = "midi_note_tokens.txt"

# Load the MIDI
midi = MidiFile(input_path)

# Open output file for writing
with open(output_path, "w") as f:
    for i, track in enumerate(midi.tracks):
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                time = msg.time
                note = msg.note
                velocity = msg.velocity
                f.write(f"{msg.type.upper()}_{note} VELOCITY_{velocity} TIME_{time}\n")

print(f"Tokenized MIDI events saved to: {output_path}")
