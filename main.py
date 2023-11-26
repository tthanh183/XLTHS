#11/26/2023 14:18 PM
import numpy as np
import os
import librosa as lrs
import librosa.display as lrsdis
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Constants
sample_rate = 16000
vowels = ['a', 'e', 'i', 'o', 'u']
NFFT = 2048
frame_size = int(20 * sample_rate / 1000)  # Frame length in samples
hop_size = int(10 * sample_rate / 1000)  # Frame step in samples


def extract_middle_part(signal):
    length = len(signal)
    part_len = length // 3
    signal = signal[part_len :part_len * 2]
    return signal


def custom_frame(signal, frame_length, hop_size):
    num_samples = len(signal)
    num_frames = 1 + (num_samples - frame_length) // hop_size
    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_length
        frames[i] = signal[start:end]

    return frames.T  # Return frames array, each frame element is horizontal


def calculate_mean_fft(signal, frame_length=frame_size, hop_size=hop_size, NFFT=NFFT):
    middle_part = extract_middle_part(signal)
    frames = custom_frame(middle_part, frame_length, hop_size)

    # Perform FFT on each frame
    fft_frames = np.fft.fft(frames, NFFT, axis=0)

    # Get the absolute values of the real and imaginary components
    abs_fft_frames = np.abs(fft_frames)

    # Calculate the mean value across frames horizontally
    mean_abs_fft_frames = np.mean(abs_fft_frames, axis=1)

    return mean_abs_fft_frames


def calculate_each_vowel_feature(signals):
    temp = []
    for signal in signals:
        temp.append(calculate_mean_fft(signal))
    temp = np.array(temp)
    feature = np.mean(temp, axis=0)
    return feature


def read_vowels_file(data_path, target_vowel):
    vowel_signals = []

    for speaker_dir in sorted(os.listdir(data_path)):
        speaker_path = os.path.join(data_path, speaker_dir)

        for file_name in sorted(os.listdir(speaker_path)):
            if file_name[0] == target_vowel:
                file_path = os.path.join(speaker_path, file_name)
                signal, fs = lrs.load(file_path)
                vowel_signals.append(signal)

    return vowel_signals


def predict_signal_label(signal, known_features, target_labels):
    min_distance = 1e9
    predicted_label = None

    test_feature = calculate_mean_fft(signal)

    for label, feature in zip(target_labels, known_features):
        distance = np.linalg.norm(test_feature - feature)
        if distance < min_distance:
            min_distance = distance
            predicted_label = label
    return predicted_label


def evaluate_testing_set(testing_path, known_features, target_labels):
    correct_predictions = 0
    total_files = 0

    for dirs in sorted(os.listdir(testing_path)):
        sub_path = os.path.join(testing_path, dirs)
        for file_name in sorted(os.listdir(sub_path)):
            file_path = os.path.join(sub_path, file_name)
            true_label = file_name[0]
            signal, fs = lrs.load(file_path)

            predicted_label = predict_signal_label(signal, known_features, target_labels)
            print(sub_path)
            print(f"File: {file_name} - True: {true_label}, Predicted: {predicted_label}")

            correct_predictions += true_label == predicted_label
            total_files += 1

    accuracy = (correct_predictions / total_files) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


# Paths
training_path = "D:\\ex2\\trimmed-NguyenAmHuanLuyen-16k"
testing_path = "D:\\ex2\\trimmed-NguyenAmKiemThu-16k"



# Extract features
features = []
for i in range(5):
    signals = read_vowels_file(training_path, vowels[i])
    features.append(calculate_each_vowel_feature(signals))

# Print features
for i in range(5):
    print(vowels[i], ": ", features[i])

# Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, ...,/u/), Đúng/Sai 
# (dựa vào nhãn tên file).
# Evaluate testing set
target_labels = vowels
evaluate_testing_set(testing_path, features, target_labels)


# # Xuất 05 vector đặc trưng biểu diễn 05 nguyên âm trên cùng 01 đồ thị.
# # Plot features
# frequencies = np.fft.fftfreq(NFFT, 1/sample_rate)

# for i in range(5):
#     plt.plot(frequencies, features[i], label=vowels[i])

# plt.title('Vector dac trung cua 5 nguyen am')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()



def create_confusion_matrix(testing_path, known_features, target_labels):
    true_labels = []
    predicted_labels = []

    for dirs in sorted(os.listdir(testing_path)):
        sub_path = os.path.join(testing_path, dirs)
        for file_name in sorted(os.listdir(sub_path)):
            file_path = os.path.join(sub_path, file_name)
            true_label = file_name[0]
            signal, fs = lrs.load(file_path)

            predicted_label = predict_signal_label(signal, known_features, target_labels)

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

    # Tạo confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=target_labels)

    # Hiển thị confusion matrix bằng heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_labels, yticklabels=target_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Đường dẫn của thư mục chứa 21 tập folder
testing_path = "D:/ex2/trimmed-NguyenAmKiemThu-16k"

# Tạo confusion matrix
create_confusion_matrix(testing_path, features, vowels)



