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

    def compute_spectrogram(self, mag=False, gamma=0, N=1024, H=512):
        X = librosa.stft(self.audio, pad_mode='constant', center=True)
        if mag:
            X = np.abs(X)**2
            if gamma > 0:
                X = np.log(1 + gamma*X)

        T_coef = np.arange(X.shape[1])*H/self.fs

        K = N//2
        F_coef = np.arange(K + 1)*self.fs/N

        return (X, T_coef, F_coef)
    
    def f_pitch(self, p, pitch_ref=69, freq_ref=440.0):
        return (2**((p - pitch_ref)/12)*freq_ref)
    
    def pool_pitch(self, p, N, pitch_ref=69, freq_ref=440.0):
        lower = self.f_pitch(p - 0.5, pitch_ref, freq_ref)
        upper = self.f_pitch(p + 0.5, pitch_ref, freq_ref)

        k = np.arange(N//2 + 1)
        k_freq = k*self.fs/N # F_coef(k, Fs, N)
        mask = np.logical_and(lower <= k_freq, k_freq < upper)

        return k[mask]
    
    def compute_spec_log_freq(self, gamma=0, N=2048, H=512):
        X, _, _ = self.compute_spectrogram(mag=True, gamma=gamma, N=N, H=H)

        X_LF = np.zeros((128, X.shape[1]))
        for p in range(128):
            k = self.pool_pitch(p, N)
            X_LF[p, :] = X[k, :].sum(axis=0)

        F_coef_pitch = np.arange(128)
        
        return X_LF, F_coef_pitch
    
    def compute_chromagram(self, gamma=0, N=2048, H=512):
        X_LF, _ = self.compute_spec_log_freq(gamma=gamma, N=N, H=H)

        C = np.zeros((12, X_LF.shape[1]))
        p = np.arange(128)

        for c in range(12):
            mask = (p%2) == c
            C[c, :] = X_LF[mask, :].sum(axis=0)

        return C
    
    def compute_MFCC(self, N=2048, H=512):
        return librosa.feature.mfcc(y=self.audio, sr=self.fs, hop_length=H, n_fft=N)

    def plot_spectrogram(self, X, T_coef, F_coef):
        # Plot spectrogram
        plt.figure(figsize=(10, 4))
        extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
        plt.imshow(X, aspect='auto', origin='lower', extent=extent)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(0, 2000)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def spectral_gating(self, N=2048, H=512, thresh=2.25):
        # Implement Adaptive Spectral Gating (if spec_gating=True)
        noise_seg = self.audio[:int(0.5*self.fs)]
        X_noise = librosa.stft(noise_seg, pad_mode='constant', center=True)

        # Calculate the median and standard deviation of the noise power spectrum
        noise_median = np.median(np.abs(X_noise), axis=1)
        noise_std = np.std(np.abs(X_noise), axis=1)

        # Create the dynamic mask based on the noise profile statistics
        adaptive_threshold = noise_median + thresh*noise_std
        adaptive_threshold = median_filter(adaptive_threshold, size=3) # Smooth the threshold

        # Apply the adaptive threshold to the STFT
        X, _, _ = self.compute_spectrogram()
        mask = np.abs(X) > adaptive_threshold[:, None]
        X_denoised = X*mask
        X_denoised = np.log(1 + 100*np.abs(X_denoised)**2)

        T_coef = np.arange(X_denoised.shape[1])*H/self.fs

        K = N//2
        F_coef = np.arange(K + 1)*self.fs/N

        return (X_denoised, T_coef, F_coef)

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