#11/24/2023 11:27 PM
import os
import numpy as np
import librosa
import soundfile as sf

sample_rate = 16000
def calculate_ste(frame):

    # Tính Short-Time Energy (STE) cho một frame
    return np.sum(np.square(frame))

def trim_silence_at_ends(audio, frame_length=0.025, min_silence=0.1):
    # Tính toán kích thước frame và số lượng frame
    frame_size = int(frame_length * sample_rate)
    num_frames = len(audio) // frame_size

    # Tính STE cho từng frame
    STEs = np.array([calculate_ste(audio[i*frame_size:(i+1)*frame_size]) for i in range(num_frames)])

    # Xác định ngưỡng để phát hiện âm thanh yên tĩnh
    avg_STE = np.average(STEs)
    threshold = 0.1 * avg_STE

    # Xác định frame yên tĩnh ở đầu file
    start_index = 0
    while STEs[start_index] < threshold:
        start_index += 1

    # Xác định frame yên tĩnh ở cuối file
    end_index = num_frames - 1
    while STEs[end_index] < threshold:
        end_index -= 1

    # Lấy khoảng âm thanh không yên tĩnh
    trimmed_audio = audio[start_index*frame_size:(end_index+1)*frame_size]

    return trimmed_audio

def signal_trimming(file_path, trim_silence=True, frame_length=0.025, min_silence=0.1):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Loại bỏ khoảng yên tĩnh ở đầu và cuối nếu cần
    if trim_silence:
        audio = trim_silence_at_ends(audio, frame_length, min_silence)

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
            trimmed_audio, sample_rate = signal_trimming(exact_input_path)

            # Ghi âm thanh đã được trim ra file đầu ra
            sf.write(exact_output_path, trimmed_audio, sample_rate)
            print(exact_output_path)

# Đường dẫn của thư mục chứa 21 tập folder
main_path = "D:/last_hope/NguyenAmKiemThu-16k"

# Đường dẫn thư mục đầu ra
output_path = "D:/last_hope/trimmed-NguyenAmKiemThu-16k"

# # Đường dẫn của thư mục chứa 21 tập folder
# main_path = "D:/last_hope/NguyenAmHuanLuyen-16k"

# # Đường dẫn thư mục đầu ra
# output_path = "D:/last_hope/trimmed-NguyenAmHuanLuyen-16k"
# Xử lý trên toàn bộ cây thư mục
browse_files(main_path, output_path)
