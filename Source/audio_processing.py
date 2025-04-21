import os, sys
import numpy as np
from numba import jit
import librosa
from matplotlib import pyplot as plt
from scipy.signal import stft, istft
from scipy.ndimage import median_filter
from scipy import ndimage

class Audio_Processing:
    def __init__(self, audio, fs=22050):
        self.audio = audio
        self.fs = fs

    def get_spectrogram(self, plot_spec=False, spec_gating=False, N=2048, H=512, thresh=2.25):
        X = librosa.stft(self.audio, pad_mode='constant', center=True)
        X = np.log(1 + 100*np.abs(X)**2)

        # Plot (if plot_spec=True)
        if plot_spec:
            T_coef = np.arange(X.shape[1])*H/self.fs

            K = N//2
            F_coef = np.arange(K + 1)*self.fs/N

            plt.figure(figsize=(10, 4))
            extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
            plt.imshow(X, aspect='auto', origin='lower', extent=extent)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # Implement Adaptive Spectral Gating (if spec_gating=True)
        if spec_gating:
            noise_seg = self.audio[:int(0.5*self.fs)]
            X_noise = librosa.stft(noise_seg, pad_mode='constant', center=True)

            # Calculate the median and standard deviation of the noise power spectrum
            noise_median = np.median(np.abs(X_noise), axis=1)
            noise_std = np.std(np.abs(X_noise), axis=1)

            # Create the dynamic mask based on the noise profile statistics
            adaptive_threshold = noise_median + thresh*noise_std
            adaptive_threshold = median_filter(adaptive_threshold, size=3) # Smooth the threshold

            # Apply the adaptive threshold to the STFT
            mask = np.abs(X) > adaptive_threshold[:, None]
            X_denoised = X*mask
            X_denoised = np.log(1 + 100*np.abs(X_denoised)**2)

            T_coef = np.arange(X_denoised.shape[1])*H/self.fs

            K = N//2
            F_coef = np.arange(K + 1)*self.fs/N

            return (X, X_denoised, T_coef, F_coef)
        
        return X

    # @jit(nopython=True)
    def get_MFCC(self, N, H):
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.fs, n_mfcc=N)
        t = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=self.fs, hop_length=H)

        return mfccs, t
    
    # @jit(nopython=True)
    def plot_MFCC(self, N, H):
        mfccs, t = self.get_MFCC(N, H)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=self.fs, x_axis='time', y_axis='linear', cmap='viridis')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.title('MFCC Spectrogram')
        plt.ylim(0, 1200)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()