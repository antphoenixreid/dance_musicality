import os, sys
import numpy as np
from numba import jit
import librosa
from matplotlib import pyplot as plt
from scipy.signal import stft, istft, find_peaks, get_window
from scipy.ndimage import median_filter
from scipy import ndimage, fftpack

class Audio_Processing:
    # Set the Initial Variables (audio, Sample Frequency, Number of FFT, and Hopsize)
    def __init__(self, audio, fs=22050, N=1024, H=512, spec_gate=False, thresh=2.0):
        self.audio = audio
        self.fs = fs
        self.N = N
        self.H = H

        # (optional) Apply Spectral Gating
        if spec_gate:
            self.audio = self.spectral_gating(thresh=thresh)

    # Internal STFT
    def __stft(self, win_func='hann'):
        # Zero-padding if signal is shorter than one frame
        if len(self.audio) < self.N:
            audio = np.pad(self.audio, (0, self.N - len(self.audio)), mode='constant')

        # Total number of frames
        num_frames = 1 + (len(audio) - self.N)//self.H
        
        # Prepare Window
        window = get_window(win_func, self.N)

        # Allocate STFT Matrix
        stft_matrix = np.empty((self.N//2 + 1, num_frames), dtype=np.complex64)

        # Compute STFT
        for i in range(num_frames):
            start = i*self.H
            frame = audio[start:start + self.N]
            windowed = frame*window
            spectrum = np.fft.rfft(windowed)
            stft_matrix[:, i] = spectrum

        # Frequency and Time Axes
        F_coef = np.fft.rfftfreq(self.N, d=1/self.fs)
        T_coef = np.arange(num_frames)*self.H/self.fs

        return stft_matrix, F_coef, T_coef

    def compute_spectrogram(self, mag=False, gamma=0):
        X, _, _ = self._stft()
        if mag:
            X = np.abs(X)**2
            if gamma > 0:
                X = np.log(1 + gamma*X)

        T_coef = np.arange(X.shape[1])*self.H/self.fs

        K = self.N//2
        F_coef = np.arange(K + 1)*self.fs/self.N

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
    
    def compute_spec_log_freq(self, gamma=100):
        X, _, _ = self.compute_spectrogram(mag=True, gamma=gamma)

        X_LF = np.zeros((128, X.shape[1]))
        for p in range(128):
            k = self.pool_pitch(p, self.N)
            X_LF[p, :] = X[k, :].sum(axis=0)

        F_coef_pitch = np.arange(128)
        
        return X_LF, F_coef_pitch
    
    def compute_chromagram(self, gamma=100):
        X_LF, _ = self.compute_spec_log_freq(gamma=gamma)

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
    
    def compute_MFCC(self, num_mfcc=13, num_filters=26, f_min=0, f_max=None):
        if f_max is None:
            f_max = self.fs/2

        # Pre-emphasis (optional, slight boost of high frequencies)
        emphasized_audio = np.append(self.audio[0], self.audio[1:] - 0.97*self.audio[:-1])

        # Framing
        num_frames = 1 + int((len(emphasized_audio) - self.N)/self.H)
        frames = np.zeros((num_frames, self.N))

        for i in range(num_frames):
            start = i*self.H
            frames[i] = emphasized_audio[start:start + self.N]

        # Windowing
        frames *= np.hamming(self.N)

        # Fourier Transform and Power Spectrum
        mag_frames = np.abs(np.fft.rfft(frames, self.N)) # Magnitude of FFT
        pow_frames = (mag_frames**2)/self.N # Power Spectrum

        # Mel Filter Bank
        mel_filters = self.get_mel_filterbank(num_filters, N=N, f_min=f_min, f_max=f_max)
        mel_energy = np.dot(pow_frames, mel_filters.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy) # Numerical Stability

        # Log Mel Spectrum 
        log_mel_energy = np.log(mel_energy)

        mfccs = fftpack.dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, :num_mfcc]

        return mfccs.T
    
    def onset_peak(self):
        frames = []
        for i in range(0, len(self.audio) - self.N, self.H):
            frame = self.audio[i:i + self.N]
            energy = np.sum(frame**2)
            frames.append(energy)

        envelope = np.array(frames)
        onset_env = np.diff(envelope)
        onset_env = np.maximum(onset_env, 0)

        return onset_env

    def compute_tempogram(self, min_bpm=30, max_bpm=300):
        # Get Onset Envelope
        onset_envelope = self.onset_peak()
        
        # Paramters
        win_size = 384
        half_window = win_size//2

        # Pad Onset Envelope
        onset_padded = np.pad(onset_envelope, (half_window, half_window), mode='constant')

        # Setup BPM Envelope
        max_lag = int((60/min_bpm)*self.fs/self.H)
        min_lag = int((60/max_bpm)*self.fs/self.H)
        tempo_axis = 60*self.fs/(self.H*np.arange(min_lag, max_lag))

        tempogram = []

        for i in range(half_window, len(onset_padded) - half_window):
            frame = onset_padded[i - half_window:i + half_window]
            ac = np.correlation(frame, frame, mode='full')
            ac = ac[ac.size//2:] # Keep only positive lags
            ac = ac[min_lag:max_lag] # Keep BPM relevant lags

            if np.max(ac) > 0:
                ac = ac/np.max(ac) # Normalize
            tempogram.append(ac)

        tempogram = np.array(tempogram).T # (lag x time frames)

        return tempo_axis, tempogram

    def spectral_gating(self, thresh=2.25):
        # Implement Adaptive Spectral Gating (if spec_gating=True)
        noise_seg = self.audio[:int(0.5*self.fs)]
        X_noise = self.stft(noise_seg, pad_mode='constant', center=True)

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

        T_coef = np.arange(X_denoised.shape[1])*self.H/self.fs

        K = self.N//2
        F_coef = np.arange(K + 1)*self.fs/self.N

        cleaned_audio = istft(X_denoised)

        return cleaned_audio