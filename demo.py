#11/24/2023 11:27 PM
import numpy as np
import os
import scipy.io.wavfile as wav

from scipy.io.wavfile import read, write
import librosa as lrs
import librosa.display as lrsdis
import matplotlib.pyplot as plt

Fs = 16000 # Tần số lấy mẫu của audio, luôn là 16000
char = ['a','e','i','o','u']
#Tham số 
NFFT = 512 # độ dài vector FFT
# frame_size = (int)(Fs*0.03) # Kích thước một frame
# hop_size = (int)(Fs*0.02) # Kích thước một bước nhảy
frame_size = int(20 * Fs / 1000)  # Frame length in samples
hop_size = int(10 * Fs / 1000)  # Frame step in samples
training_path = "D:\\last_hope\\trimmed-NguyenAmHuanLuyen-16k" # đường dẫn đến file training
testing_path = "D:\\last_hope\\trimmed-NguyenAmKiemThu-16k"    # đường dẫn đến file test

def Divide(audio):
    length = len(audio)
    part_len = length // 3
    audio = audio[part_len+1:part_len*2]
    return audio

# def TrimPart(audio):
#     length = len(audio)
#     expect_len = length//4
#     start_id = expect_len//3
#     audio = audio[start_id:start_id+expect_len]
#     return audio

def ReadFileByName(inputPath, r):
    audios = []
    # trả về danh sách các bản ghi âm đã cắt của ký tự char[r]
    for dirs in sorted(os.listdir(inputPath)):
        sub_path = os.path.join(inputPath, dirs)
        for files in sorted(os.listdir(sub_path)):
            if(files[0]!=char[r]):
                continue
            audio_path = os.path.join(sub_path, files)
            # print(audio_file)
            audio, fs = lrs.load(audio_path)
            # Tách audio và thêm vào danh sách audios
            audios.append((audio))
    return audios

def cal_nfft(audio): #lấy giá trị vector FFT của một audio
    # middle_part = TrimPart(audio=audio)
    frames = lrs.util.frame(audio, frame_length=frame_size, hop_length=hop_size)
    fft_frames = np.fft.fft(frames,NFFT, axis=0) # hiện tại một cột là một vector
    res = fft_frames
    res = np.abs(res)#lấy biên độ là số dương
    res = np.mean(res,axis=1)
    return res

def cal_feature(audios,c):
    # trích xuất đặc trưng từ danh sách các audio
    # lấy trung bình
    temp = [] #mảng hai chiều chứa thông tin các vector 
    frequencies = np.fft.fftfreq(NFFT, 1/Fs)
    for audio in audios:
        temp.append(cal_nfft(audio))
        plt.plot(frequencies,temp[-1])
    temp = np.array(temp)
    if (len(temp)!=21): print(len(temp))
    feature = np.mean(temp,axis = 0)    
    plt.plot(frequencies, feature)
    plt.title('Biểu đồ Phổ Tần Số '+char[c])
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Biên độ')
    return feature


# Tất cả các bản ghi đều có tần số lấy mẫu là 16kHz
features = [] # Kết quả , features[0] tương ứng vector FFT(ndarray) của chữ a, 1 của e,...
for i in range(0,5):
    # if (i!=1): continue
    audios = ReadFileByName(training_path,i)
    features.append(cal_feature(audios,i))
# print(features)
for i in range(0,5):
    print(char[i],": ",features[i])

# Hàm để kiểm tra một audio file
def predict_audio_label(audio, known_features, target_labels):
    min_distance = float('inf')
    predicted_label = None
    # Tính đặc trưng của file kiểm tra
    test_feature = cal_nfft(audio)
    # So sánh với đặc trưng đã biết
    for label, feature in zip(target_labels, known_features):
        distance = np.linalg.norm(test_feature - feature)
        if distance < min_distance:
            min_distance = distance
            predicted_label = label
    return predicted_label
# Hàm để kiểm tra toàn bộ tập dữ liệu kiểm tra
def evaluate_testing_set(testing_path, known_features, target_labels):
    correct_predictions = 0
    total_files = 0
    # Duyệt qua từng file trong tập kiểm tra
    for dirs in sorted(os.listdir(testing_path)):
        sub_path = os.path.join(testing_path, dirs)
        for file_name in sorted(os.listdir(sub_path)):
            file_path = os.path.join(sub_path, file_name)
            true_label = file_name[0]  # Giả định nhãn đúng là ký tự đầu tiên trong tên file
            audio, fs = lrs.load(file_path)
            # Dự đoán nhãn và kiểm tra
            predicted_label = predict_audio_label(audio, known_features, target_labels)
            print(sub_path)
            print(f"File: {file_name} - True: {true_label}, Predicted: {predicted_label}")
            correct_predictions += true_label == predicted_label
            total_files += 1
    # Tính tỷ lệ chính xác
    accuracy = (correct_predictions / total_files) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
# Khởi tạo nhãn dự đoán và features đã biết
target_labels = char  # ['a', 'e', 'i', 'o', 'u']
# Đánh giá tập kiểm tra
evaluate_testing_set(testing_path, features, target_labels)





