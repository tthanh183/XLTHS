import numpy as np
import os
from scipy.io.wavfile import read, write
import librosa as lrs
import librosa.display as lrsdis
import matplotlib.pyplot as plt
#Const 
Fs = 16000 # Tần số lấy mẫu của audio, luôn là 16000
char = ['a','e','i','o','u']
#Tham số 
NFFT = 2048 # độ dài vector FFT
frame_size = (int)(Fs*0.03) # Kích thước một frame
hop_size = (int)(Fs*0.02) # Kích thước một bước nhảy
# training_path = ".\\trimmed_signals\\HuanLuyen-16k" # đường dẫn đến file training
# testing_path = ".\\trimmed_signals\\KiemThu-16k"
training_path = "D:\\last_hope\\trimmed-NguyenAmHuanLuyen-16k" # đường dẫn đến file training
testing_path = "D:\\last_hope\\NguyenAmKiemThu-16k"
def TrimPart(audio):
    length = len(audio)
    expect_len = length//4
    start_id = expect_len//3
    audio = audio[start_id:start_id+expect_len]
    return audio

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
            audios.append(TrimPart(audio))
    return audios
def cal_nfft(audio): #lấy giá trị vector FFT của một audio
    frames = lrs.util.frame(audio, frame_length=frame_size, hop_length=hop_size)
    fft_frames = np.fft.fft(frames,NFFT, axis=0) # hiện tại một cột là một vector
    res = fft_frames
    res = np.abs(res)#lấy biên độ là số dương
    res = np.mean(res,axis=1)
    # print(len(res[0]))
    # frequencies = np.fft.fftfreq(NFFT, 1/Fs)
    # plt.plot(frequencies, res)
    # plt.title('Biểu đồ Phổ Tần Số')
    # plt.xlabel('Tần số (Hz)')
    # plt.ylabel('Biên độ')
    # plt.show()
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
print(features)
cnt = 0
def check_feature(files_path, val):
    audios = ReadFileByName(files_path,val)
    cnt = 0
    cval= [0,0,0,0,0]
    for audio in audios:
        decision = -1
        min_dif = 1e18
        i =-1
        fft_audio = cal_nfft(audio)
        for feature in features:
            i+=1
            temp = np.linalg.norm(fft_audio-feature)
            if (temp<=min_dif):
                decision = i
                min_dif = temp
        # print("Testing "+char[val]+", decision "+char[decision])
        cval[decision]+=1
        if val == decision:
            cnt+=1
        #return only the character which is most similar
    print("Testing "+char[val]+": ")
    pn = ""
    for i in range(0,5):
        pn+=("| = "+char[i]+": "+str(round(100*cval[i]/21,2))+"% |")
    print(pn)
    return cnt

res =0
for i in range(0,5):
    res+=check_feature(testing_path,i)
print(str(res)+"/105")