import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

def calculate_ste(frame):
    # Tính Short-Time Energy (STE)
    return np.sum(np.square(frame))

def plot_waveforms(original_audio, trimmed_audio, sample_rate):
    # Tạo trục thời gian cho âm thanh gốc
    time_original = np.arange(0, len(original_audio)) / sample_rate

    # Tạo trục thời gian cho âm thanh được cắt ra
    time_trimmed = np.arange(0, len(trimmed_audio)) / sample_rate

    # Vẽ âm thanh gốc
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_original, original_audio, label='Original Audio')
    plt.title('Original Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Vẽ âm thanh được cắt ra
    plt.subplot(2, 1, 2)
    plt.plot(time_trimmed, trimmed_audio, label='Trimmed Audio')
    plt.title('Trimmed Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Hiển thị đồ thị
    plt.tight_layout()
    plt.show()

def trim_audio(file_path, frame_length=0.025):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Tính kích thước của khung và số lượng khung
    frame_size = int(frame_length * sample_rate)
    num_frames = len(audio) // frame_size

    # Tính STE cho từng khung
    STEs = np.array([calculate_ste(audio[i * frame_size:(i + 1) * frame_size]) for i in range(num_frames)])

    # Xác định ngưỡng để phát hiện âm thanh yên tĩnh
    avg_STE = np.average(STEs)
    threshold = 0.1 * avg_STE

    # Xác định khung âm thanh yên tĩnh ở đầu tệp
    start_index = 0
    while STEs[start_index] < threshold:
        start_index += 1

    # Xác định khung âm thanh yên tĩnh ở cuối tệp
    end_index = num_frames - 1
    while STEs[end_index] < threshold:
        end_index -= 1

    # Lấy phần âm thanh không yên tĩnh
    trimmed_audio = audio[start_index * frame_size:(end_index + 1) * frame_size]

    # Vẽ đồ thị âm thanh
    plot_waveforms(audio, trimmed_audio, sample_rate)

    return trimmed_audio, sample_rate

# Đường dẫn của thư mục chứa 21 tập folder
main_path = "D:/ex2/NguyenAmHuanLuyen-16k"

# Lấy mẫu một file để minh họa
sample_file_path = os.path.join(main_path, '23MTL', 'a.wav')

# Đường dẫn thư mục đầu ra
output_path = "D:/ex2/trimmed-NguyenAmHuanLuyen-16k"

# Trim âm thanh và vẽ đồ thị
trim_audio(sample_file_path)
