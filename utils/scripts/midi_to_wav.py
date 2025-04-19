import fluidsynth

def midi_to_wav(midi_file, wav_file, soundfont="example.sf2"):
    fs = fluidsynth.Synth()
    fs.start(driver="file")
    sfid = fs.sfload(soundfont)
    fs.program_select(0, sfid, 0, 0)
    fs.midi_file_play(midi_file)
    fs.write_wav(wav_file)
    fs.delete()
#Example Use
midi_to_wav("data\generated\generated_midis\chant_triplet_output_diatonic.mid", "data\generated\generated_wavs\chant_triplet_output_diatonic.wav", soundfont="FluidR3_GM.sf2")