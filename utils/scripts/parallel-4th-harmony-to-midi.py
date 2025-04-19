from mido import MidiFile, MidiTrack, Message

# Replace this with your actual melody triplets
melody_triplets = [
    [53, 70, 240], [55, 65, 192], [57, 60, 160], [59, 60, 160],
    [60, 64, 160], [62, 66, 160], [64, 68, 160], [65, 66, 192],
    [67, 70, 160], [69, 72, 160], [71, 68, 160], [72, 70, 240],
    [71, 65, 192], [69, 66, 160], [67, 68, 160], [65, 64, 160],
    [64, 66, 160], [62, 65, 192], [60, 63, 160], [59, 62, 240]
]

INTERVAL = 5  # Perfect fourth

mid = MidiFile()
melody_track = MidiTrack()
harmony_track = MidiTrack()
mid.tracks.append(melody_track)
mid.tracks.append(harmony_track)

for note, velocity, duration in melody_triplets:
    # Melody
    melody_track.append(Message("note_on", note=note, velocity=velocity, time=0))
    melody_track.append(Message("note_off", note=note, velocity=0, time=duration))

    # Harmony (perfect 4th above)
    harmony_note = note + INTERVAL
    harmony_track.append(Message("note_on", note=harmony_note, velocity=velocity, time=0))
    harmony_track.append(Message("note_off", note=harmony_note, velocity=0, time=duration))

mid.save("chant_parallel_organum_4ths.mid")
print("✅ Saved as chant_parallel_organum_4ths.mid")