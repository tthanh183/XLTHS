import os
import numpy as np
import librosa
import soundfile as sf

sample_rate = 16000
def calculate_ste(frame):
    # Tính Short-Time Energy (STE) 
    return np.sum(np.square(frame))

def trim_audio(file_path, frame_length=0.025):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Tính kích thước của khung và số lượng khung
    frame_size = int(frame_length * sample_rate)
    num_frames = len(audio) // frame_size

    # Tính STE cho từng khung
    STEs = np.array([calculate_ste(audio[i*frame_size:(i+1)*frame_size]) for i in range(num_frames)])

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
    audio = audio[start_index*frame_size:(end_index+1)*frame_size]

    return audio, sample_rate



def browse_files(signal_path, output_path):
    for file in sorted(os.listdir(signal_path)):
        sub_path = os.path.join(signal_path, file)
        sub_output_path = os.path.join(output_path, file)

        # Tạo thư mục đầu ra nếu nó chưa tồn tại
        os.makedirs(sub_output_path, exist_ok=True)

        for sub_file in sorted(os.listdir(sub_path)):
            exact_input_path = os.path.join(sub_path, sub_file)
            exact_output_path = os.path.join(sub_output_path, sub_file)

            # Lấy âm thanh đã được trim, mặc định là trim cả yên tĩnh ở đầu và cuối
            trimmed_audio, sample_rate = trim_audio(exact_input_path)

            # Ghi âm thanh đã được trim ra file đầu ra
            sf.write(exact_output_path, trimmed_audio, sample_rate)
            print(exact_output_path)

# Đường dẫn của thư mục chứa 21 tập folder
main_path1 = "D:/ex2/NguyenAmKiemThu-16k"

# Đường dẫn thư mục đầu ra
output_path1 = "D:/ex2/trimmed-NguyenAmKiemThu-16k"
browse_files(main_path1, output_path1)

# Đường dẫn của thư mục chứa 21 tập folder
main_path = "D:/ex2/NguyenAmHuanLuyen-16k"

# Đường dẫn thư mục đầu ra
output_path = "D:/ex2/trimmed-NguyenAmHuanLuyen-16k"
# Xử lý trên toàn bộ cây thư mục
browse_files(main_path, output_path)
