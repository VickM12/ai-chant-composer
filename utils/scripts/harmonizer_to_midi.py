
from mido import MidiFile, MidiTrack, Message

def save_duet_to_midi(melody_triplets, harmony_triplets, output_path="harmonized_chant.mid"):
    mid = MidiFile()
    melody_track = MidiTrack()
    harmony_track = MidiTrack()
    mid.tracks.append(melody_track)
    mid.tracks.append(harmony_track)

    for m_triplet, h_triplet in zip(melody_triplets, harmony_triplets):
        m_note, m_vel, m_dur = m_triplet
        h_note, h_vel, h_dur = h_triplet

        if m_note > 0:
            melody_track.append(Message("note_on", note=m_note, velocity=m_vel, time=0))
            melody_track.append(Message("note_off", note=m_note, velocity=0, time=m_dur))
        else:
            melody_track.append(Message("note_on", note=0, velocity=0, time=m_dur))

        if h_note > 0:
            harmony_track.append(Message("note_on", note=h_note, velocity=h_vel, time=0))
            harmony_track.append(Message("note_off", note=h_note, velocity=0, time=h_dur))
        else:
            harmony_track.append(Message("note_on", note=0, velocity=0, time=h_dur))

    mid.save(output_path)
    print(f"🎵 Saved duet to {output_path}")
