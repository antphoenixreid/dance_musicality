import os, sys
import numpy as np
from numba import jit
import librosa
from matplotlib import pyplot as plt

from audio_processing import Audio_Processing

def main():
    audio_file = 'D:/Engineering/Signal Processing/Personal Projects/Dance Musicality/Data/Red Bull Dance Your Style 2022/Sara/Sara_clip.mp3'
    x, Fs = librosa.load(audio_file)

    if len(x.shape) == 2:
        x = librosa.to_mono(x)

    processed_audio = Audio_Processing(x, fs=Fs)
    (X, T_coef, F_coef) = processed_audio.compute_spectrogram(mag=True, gamma=100)

    # processed_audio.plot_spectrogram(X, T_coef, F_coef)

    # (X_denoised, T_coef, F_coef) = processed_audio.spectral_gating(thresh=1.5)
    # processed_audio.plot_spectrogram(X_denoised, T_coef, F_coef)

    processed_audio.visualize_feature()

if __name__ == "__main__":
    main()