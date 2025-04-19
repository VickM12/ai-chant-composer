from mido import Message, MidiFile, MidiTrack

# Load tokens from the file
with open("generated_chant_tokens.txt", "r") as f:
    tokens = [line.strip() for line in f if line.strip()]

# Group into full events safely
events = []
i = 0
while i < len(tokens) - 2:
    if tokens[i].startswith("NOTE_") and tokens[i+1].startswith("VELOCITY_") and tokens[i+2].startswith("TIME_"):
        events.append((tokens[i], tokens[i+1], tokens[i+2]))
        i += 3
    else:
        i += 1  # Skip malformed data

# Create a new MIDI file
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for note_token, velocity_token, time_token in events:
    try:
        note_type, note = note_token.rsplit("_", 1)
        velocity = int(velocity_token.split("_")[1])
        time = int(time_token.split("_")[1])
        track.append(Message(note_type.lower(), note=int(note), velocity=velocity, time=time))
    except Exception as e:
        print(f"Skipping invalid group: {note_token}, {velocity_token}, {time_token} ({e})")

# Save MIDI file
mid.save("generated_chant_output.mid")
print("✅ Clean MIDI file saved as 'generated_chant_output.mid'")
