import cv2
import torch
import numpy as np
import queue
import threading
import time
import audio
from hparams import hparams as hp
from models import Wav2Lip
import face_detection
from scipy.io.wavfile import write, read
import sounddevice as sd

# --- CẤU HÌNH ---
CHECKPOINT_PATH = 'checkpoints/wav2lip_gan.pth' # Đường dẫn model
FACE_VIDEO_PATH = 'ian5sec25fps.mp4' # Video mẫu (Avatar)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FPS = 25
SAMPLE_RATE = 16000
BUFFER_SIZE = 5 # Số frame buffer để hiển thị mượt

# --- KHỞI TẠO MODEL (Load 1 lần duy nhất) ---
print("Loading model...")
model = Wav2Lip()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=lambda storage, loc: storage)
s = checkpoint["state_dict"]
new_s = {k.replace('module.', ''): v for k, v in s.items()}
model.load_state_dict(new_s)
model = model.to(DEVICE).eval()
print("Model loaded.")

# --- TIỀN XỬ LÝ VIDEO MẪU (ONE-TIME SETUP) ---
# Thay vì load mỗi lần, ta chuẩn bị sẵn data
print("Preparing avatar data...")
video_stream = cv2.VideoCapture(FACE_VIDEO_PATH)
full_frames = []
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        break
    full_frames.append(frame)

# Phát hiện mặt 1 lần duy nhất (Optimization quan trọng nhất)
detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=DEVICE)
# Lấy frame đầu tiên để detect
face = full_frames[0]
# Chạy detect
pred = detector.get_detections_for_image(np.array(face))
if pred is None:
    raise ValueError("Không tìm thấy khuôn mặt trong video mẫu")

# Lấy tọa độ box (Cố định box cho toàn bộ video)
y1, x2, y2, x1 = pred[0] # Lưu ý thứ tự tọa độ tùy thư viện face_detection
# Thêm padding
pady1, pady2, padx1, padx2 = 0, 10, 0, 0
y1 = max(0, y1 - pady1)
y2 = min(face.shape[0], y2 + pady2)
x1 = max(0, x1 - padx1)
x2 = min(face.shape[1], x2 + padx2)

print("Avatar ready.")

# --- CÁC HÀM XỬ LÝ ---

def get_mel_chunks(audio_chunk):
    """Chuyển đổi audio chunk thành mel spectrogram"""
    # Hàm này nằm trong audio.py bạn đã gửi
    mel = audio.melspectrogram(audio_chunk)
    # Chia nhỏ mel thành các step 16 (mel_step_size)
    mel_chunks = []
    mel_idx_multiplier = 80./FPS
    
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + 16 > len(mel[0]):
            break
        mel_chunks.append(mel[:, start_idx : start_idx + 16])
        i += 1
    return mel_chunks

def datagen_single(mel_chunk, frame_idx):
    """Tạo dữ liệu đầu vào cho 1 frame duy nhất"""
    # Lấy frame tương ứng từ video mẫu (loop video)
    frame = full_frames[frame_idx % len(full_frames)]
    
    # Crop face dùng box đã fix
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (96, 96))
    
    # Chuẩn bị input mask
    img_batch = []
    mel_batch = []
    
    img_masked = face.copy()
    img_masked[96//2:] = 0 # Che nửa dưới
    
    img_batch.append(face)
    mel_batch.append(mel_chunk)
    
    img_batch = np.asarray(img_batch)
    mel_batch = np.asarray(mel_batch)
    
    img_masked = np.asarray([img_masked])
    
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
    
    return img_batch, mel_batch, frame

# --- LUỒNG XỬ LÝ (THREADING) ---

# Hàng đợi (Queue) để giao tiếp giữa các luồng
audio_queue = queue.Queue()
video_queue = queue.Queue(maxsize=BUFFER_SIZE)

def audio_thread():
    """Luồng thu âm và đưa vào queue"""
    print("Listening...")
    while True:
        # Thu âm 1 khúc ngắn (ví dụ 0.5s hoặc 1s)
        # Để realtime mượt, chunk size nhỏ tốt hơn nhưng cần cân bằng với hiệu năng
        duration = 0.5 # giây
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        
        # Xử lý nhiễu & đưa vào queue
        audio_data = recording.flatten()
        # Thêm vào queue để luồng inference xử lý
        audio_queue.put(audio_data)

def inference_thread():
    """Luồng chạy AI"""
    frame_idx = 0
    while True:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            
            # 1. Audio -> Mel
            mel_chunks = get_mel_chunks(audio_chunk)
            
            # 2. Mel -> Face
            for mel in mel_chunks:
                img_batch, mel_batch, original_frame = datagen_single(mel, frame_idx)
                
                # Chuyển sang Tensor
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(DEVICE)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(DEVICE)
                
                # Inference
                with torch.no_grad():
                    pred = model(mel_batch, img_batch)
                
                # Hậu xử lý
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255.
                pred = pred.astype(np.uint8)
                
                # Resize lại kích thước box gốc
                out_face = cv2.resize(pred, (x2-x1, y2-y1))
                
                # Ghép vào frame gốc
                result_frame = original_frame.copy()
                result_frame[y1:y2, x1:x2] = out_face
                
                # Đưa vào queue video để hiển thị
                if video_queue.full():
                    # Nếu buffer đầy, bỏ qua frame cũ để tránh lag
                    try:
                        video_queue.get_nowait()
                    except:
                        pass
                video_queue.put(result_frame)
                
                frame_idx += 1

def display_thread():
    """Luồng hiển thị video"""
    cv2.namedWindow('Realtime Avatar', cv2.WINDOW_NORMAL)
    while True:
        if not video_queue.empty():
            frame = video_queue.get()
            cv2.imshow('Realtime Avatar', frame)
        
        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == '__main__':
    t1 = threading.Thread(target=audio_thread)
    t2 = threading.Thread(target=inference_thread)
    t3 = threading.Thread(target=display_thread)
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()
