import os, sys
import numpy as np
from numba import jit
import librosa
from matplotlib import pyplot as plt
from scipy.signal import stft, istft, find_peaks
from scipy.ndimage import median_filter
from scipy import ndimage, fftpack

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
    
    def compute_spec_log_freq(self, gamma=0, N=1024, H=512):
        X, _, _ = self.compute_spectrogram(mag=True, gamma=gamma, N=N, H=H)

        X_LF = np.zeros((128, X.shape[1]))
        for p in range(128):
            k = self.pool_pitch(p, N)
            X_LF[p, :] = X[k, :].sum(axis=0)

        F_coef_pitch = np.arange(128)
        
        return X_LF, F_coef_pitch
    
    def compute_chromagram(self, gamma=0, N=1024, H=512):
        X_LF, _ = self.compute_spec_log_freq(gamma=gamma, N=N, H=H)

        C = np.zeros((12, X_LF.shape[1]))
        p = np.arange(128)

        for c in range(12):
            mask = (p%2) == c
            C[c, :] = X_LF[mask, :].sum(axis=0)

        return C
    
    def hz_to_mel(self, hz):
        return 2595*np.log10(1 + hz/700)
    
    def mel_to_hz(self, mel):
        return 700*(10**(mel/2595) - 1)
    
    def get_mel_filterbank(self, num_filters, N, f_min, f_max):
        # Generate Mel Filterbank
        low_mel = self.hz_to_mel(f_min)
        high_mel = self.hz_to_mel(f_max)
        mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
        hz_points = self.mel_to_hz(mel_points)

        bins = np.floor((N + 1)*hz_points/self.fs).astype(int)

        fbank = np.zeros((num_filters, N//2 + 1))
        for i in range(1, num_filters + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]

            for k in range(left, center):
                if center - left != 0:
                    fbank[i - 1, k] = (k - left)/(center - left)
            for k in range(center, right):
                if right - center != 0:
                    fbank[i - 1, k] = (right - k)/(right - center)

        return fbank
    
    def compute_MFCC(self, N=1024, H=512, num_mfcc=13, num_filters=26, f_min=0, f_max=None):
        if f_max is None:
            f_max = self.fs/2

        # Pre-emphasis (optional, slight boost of high frequencies)
        emphasized_audio = np.append(self.audio[0], self.audio[1:] - 0.97*self.audio[:-1])

        # Framing
        num_frames = 1 + int((len(emphasized_audio) - N)/H)
        frames = np.zeros((num_frames, N))

        for i in range(num_frames):
            start = i*H
            frames[i] = emphasized_audio[start:start + N]

        # Windowing
        frames *= np.hamming(N)

        # Fourier Transform and Power Spectrum
        mag_frames = np.abs(np.fft.rfft(frames, N)) # Magnitude of FFT
        pow_frames = (mag_frames**2)/N # Power Spectrum

        # Mel Filter Bank
        mel_filters = self.get_mel_filterbank(num_filters, N=N, f_min=f_min, f_max=f_max)
        mel_energy = np.dot(pow_frames, mel_filters.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy) # Numerical Stability

        # Log Mel Spectrum 
        log_mel_energy = np.log(mel_energy)

        mfccs = fftpack.dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, :num_mfcc]

        return mfccs.T
    
    def onset_peak(self, N=1024, H=512):
        frames = []
        for i in range(0, len(self.audio) - N, H):
            frame = self.audio[i:i + N]
            energy = np.sum(frame**2)
            frames.append(energy)

        envelope = np.array(frames)
        onset_env = np.diff(envelope)
        onset_env = np.maximum(onset_env, 0)

        return onset_env

        # plt.plot(np.arange(len(energy_diff))*H/self.fs, energy, label='Energy Change')
        # plt.plot(times, energy_diff[peaks], 'rx', label='Detected Onsets')
        # plt.title('Onset Detection (Energy-Based)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Energy Change')
        # plt.legend()
        # plt.show()

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

    def visualize_feature(self, gamma=100, N=1024, H=512):
        # Onset Detection
        onset_env = self.onset_peak(N=N, H=H)
        time_onset = np.arange(len(onset_env))*H/self.fs

        # Chroma
        C = self.compute_chromagram(gamma=gamma, N=N, H=H)
        frames_chroma = np.arange(C.shape[1])
        time_chroma = frames_chroma*H/self.fs

        # MFCC
        MFCC = self.compute_MFCC(N=N, H=H)
        frames_mfcc = np.arange(MFCC.shape[1])
        time_mfcc = frames_mfcc*H/self.fs

        # Set up the plot figure
        plt.figure(figsize=(15, 10))

        # MFCCs
        plt.subplot(3, 1, 1)
        plt.imshow(MFCC, aspect='auto', origin='lower', extent=[time_mfcc.min(), time_mfcc.max(), 0, MFCC.shape[0]])
        plt.colorbar()
        plt.title('MFCCs over Time')
        plt.ylabel('MFCC Coefficients')

        # Chroma
        plt.subplot(3, 1, 2)
        plt.imshow(C, aspect='auto', origin='lower', extent=[time_chroma.min(), time_chroma.max(), 0, 12])
        plt.colorbar()
        plt.title('Chroma Features over Time')
        plt.ylabel('Pitch Class')

        # Onset Envelope
        plt.subplot(3, 1, 3)
        plt.plot(time_onset, onset_env)
        plt.title('Onset Envelope (Energy Changes)')
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')

        plt.tight_layout()
        plt.show()

    def spectral_gating(self, N=1024, H=512, thresh=2.25):
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
    def get_MFCC(self, N=1024, H=512):
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.fs, n_mfcc=N)
        t = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=self.fs, hop_length=H)

        return mfccs, t
    
    # @jit(nopython=True)
    def plot_MFCC(self, N=1024, H=512):
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