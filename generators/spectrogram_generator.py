import os
import argparse
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(wav_path: str, output_path: str = None):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    sample_rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # Convert to mono if stereo

    plt.figure(figsize=(10, 4))
    plt.specgram(data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='inferno')
    plt.title("Spectrogram of Generated Chant")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.tight_layout()

    output_path = output_path or os.path.splitext(wav_path)[0] + "_spectrogram.png"
    plt.savefig(output_path)
    print(f"📈 Spectrogram saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a spectrogram from a WAV file.")
    parser.add_argument("--wav", type=str, required=True, help="Path to the WAV file")
    parser.add_argument("--out", type=str, help="Path to save the output PNG image")

    args = parser.parse_args()
    generate_spectrogram(args.wav, args.out)
