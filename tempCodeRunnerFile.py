input_audio_path = "D:\\trim_signal\\NguyenAmHuanLuyen-16k\\24FTL\\a.wav"
output_audio_path = "D:\\trim_signal\\trimmed-NguyenAmHuanLuyen-16k\\24FTL\\a.wav"

# Load file âm thanh
audio, sample_rate = lrs.load(input_audio_path, sr=None)
# Vẽ đồ thị âm thanh gốc và phần đã được tách
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(Divide(audio=audio))
plt.title('Original Audio')

plt.subplot(2, 1, 2)
plt.plot(audio)
plt.title('FFT of the Middle Part')


# cnt = 0
# def check_feature(files_path, val):
#     audios = ReadFileByName(files_path,val)
#     cnt = 0
#     cval= [0,0,0,0,0]
#     for audio in audios:
#         decision = -1
#         min_dif = 1e18
#         i =-1
#         fft_audio = cal_nfft(audio)
#         for feature in features:
#             i+=1
#             temp = np.linalg.norm(fft_audio-feature)
#             if (temp<=min_dif):
#                 decision = i
#                 min_dif = temp
#         # print("Testing "+char[val]+", decision "+char[decision])
#         cval[decision]+=1
#         if val == decision:
#             cnt+=1
#         #return only the character which is most similar
#     print("Testing "+char[val]+": ")
#     pn = ""
#     for i in range(0,5):
#         pn+=("| = "+char[i]+": "+str(round(100*cval[i]/21,2))+"% |")
#     print(pn)
#     return cnt

# res =0
# for i in range(0,5):
#     res+=check_feature(testing_path,i)
# print(str(res)+"/105")