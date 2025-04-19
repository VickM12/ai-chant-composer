# Gregorian Chant AI Composer

This project explores the generation of Gregorian chant using AI, with a focus on structured musical triplets (`NOTE`, `VELOCITY`, `DURATION`) and LSTM-based neural networks. The goal is to synthesize chant that adheres to diatonic structure and evokes traditional liturgical phrasing.

## Features

- Structured chant generation using a trained LSTM model
- Output in MIDI format with rests and cadence recognition
- Optional harmony generation using a secondary model
- Tools for dataset conversion and token inspection

## File Structure

- `chant_triplet_model_generator_diatonic.py` - Primary generator for chant melodies
- `harmonizer_model_trainer.py` - Trains the harmony model (optional)
- `batch_convert_chants_to_triplets.py` - Converts a folder of MIDI chants to structured triplets
- `training-sets/` - Contains `.jsonl` datasets for training
- `token-files/` - Stores generated token outputs
- `utils/` - MIDI utilities and conversion helpers

## Data Sources

Chant MIDI files were sourced from:

**Richard Lee's Gregorian Chant MIDI Archive**  
University of Arkansas  
[https://rlee.hosted.uark.edu/midi/](https://rlee.hosted.uark.edu/midi/)

These files were created using NoteWorthy Composer and are interpretive renderings of traditional Gregorian melodies.

## How to Use

1. Prepare your training data from `.mid` files using:
    ```bash
    python batch_convert_chants_to_triplets.py
    ```

2. Train the chant generator:
    ```bash
    python chant_triplet_model_trainer.py
    ```

3. Generate chant:
    ```bash
    python chant_triplet_model_generator_diatonic.py
    ```

4. (Optional) Harmonize:
    ```bash
    python chant_harmonizer_pipeline_diatonic.py
    ```

## License

MIT License — for academic and personal research. Please credit Richard Lee for the chant corpus.