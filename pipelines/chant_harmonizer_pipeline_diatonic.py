
from chant_triplet_model_generator_diatonic import generate_sequence, load_model
from harmonizer_generator import generate_harmony
from harmonizer_to_midi import save_duet_to_midi


def run_chant_harmonizer_pipeline_diatonic(
    chant_model_path="chant_triplet_model.pt",
    harmonizer_model_path="harmonizer_model_v2.pt",
    length=120,
    temperature=1.0,
    output_path="harmonized_chant_output_diatonic.mid"
):
    # Load diatonic chant model
    model = load_model(chant_model_path)

    # Seed with a modal phrase (e.g. middle C)
    seed = [[60, 64, 80]] * 10

    # Step 1: Generate melody
    melody_triplets = generate_sequence(model, seed, length)

    # Step 2: Generate harmony
    harmony_triplets = generate_harmony(
        model_path=harmonizer_model_path,
        melody_sequence=melody_triplets
    )

    # Step 3: Write to MIDI
    save_duet_to_midi(
        melody_triplets,
        harmony_triplets,
        output_path=output_path
    )

    print(f"🎼 Diatonically harmonized chant saved to: {output_path}")

if __name__ == "__main__":
    run_chant_harmonizer_pipeline_diatonic()
