import numpy as np
import librosa
import io


def extract_features_from_audio(audio_bytes, sr=22050):
    """
    Extract 59-dimensional audio features from audio bytes.
    
    Args:
        audio_bytes: Raw audio file bytes
        sr: Sample rate (default 22050 Hz)
    
    Returns:
        np.ndarray of shape (1, 59) with features:
        - 13 MFCC means
        - 13 MFCC stds
        - ZCR, Spectral Centroid, Rolloff
        - 12 Chroma means
        - 7 Spectral Contrast means
        - 6 Tonnetz means
        - Tempo
        - Spectral Bandwidth, Flatness
        - RMS, RMS std
    """
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    
    # Basic MFCC (13 dimensions × 2)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Basic spectral features (3 dimensions)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Chroma features (12 dimensions)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral contrast (7 dimensions)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Tonnetz (6 dimensions)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    # Tempo (1 dimension)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = tempo.item() if isinstance(tempo, np.ndarray) else float(tempo)
    
    # Additional spectral features (4 dimensions)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    rms_std = np.std(librosa.feature.rms(y=y))
    
    # Combine all features (59 dimensions)
    features = list(mfcc_mean) + list(mfcc_std) + [zcr, spectral_centroid, rolloff] + \
               list(chroma_mean) + list(contrast_mean) + list(tonnetz_mean) + \
               [tempo_val, spectral_bandwidth, spectral_flatness, rms, rms_std]
    
    return np.array(features).reshape(1, -1)
