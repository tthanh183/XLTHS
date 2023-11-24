import numpy as np
import librosa as lrs


# def segmentation(audio, frame_length, hop_length, energy_threshold=0.01, zc_rate_threshold=0.1):
#     frames = np.array([audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)])
#     # Calculate Short-Time Energy (STE)
#     # energy = np.sum(frames**2, axis=1)
#     energy = np.linalg.norm(frames)**2
#     energy = energy / np.max(energy)
    
#     # Calculate Zero Crossing Rate (ZCR)
#     zc_rate = np.sum(np.abs(np.diff(np.sign(frames))), axis=1) / (2 * frame_length)
#     zc_rate = zc_rate/ np.max(energy)
#     # Select frames based on energy and zc_rate thresholds
#     selected_frames = frames[(energy >= energy_threshold) & (zc_rate >= zc_rate_threshold), :]
    
#     return selected_frames

# def segmentation(audio, frame_length, hop_length, energy_threshold=0.01, zc_rate_threshold=0.1):

#     frames = np.array([audio[i:i+frame_length] for i in range(0, len(audio)-frame_length, hop_length)])
#     # Calculate Short-Time Energy (STE)
#     energy = np.sum(frames**2, axis=1)
#     energy = energy / np.max(energy)
    
#     # Calculate Zero Crossing Rate (ZCR)
#     zc_rate = np.sum(np.abs(np.diff(np.sign(frames))), axis=1) / (2 * frame_length)
#     # zc_rate = zc_rate/ np.max(energy)
    
#     # Select frames based on energy and zc_rate thresholds
#     selected_frames = frames[(energy >= energy_threshold) & (zc_rate >= zc_rate_threshold), :]
    
#     return selected_frames



