import numpy as np
import cv2

def load_data(video_path, caption_path, batch_size=32, frame_step=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    captions = open(caption_path, 'r').readlines()

    # Preprocess captions
    captions = [cap.strip() for cap in captions]

    # Load video frames in batches
    for start in range(0, frame_count, batch_size * frame_step):
        batch_frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize frames
            batch_frames.append(frame)
            for _ in range(frame_step - 1):  # Skip frames
                ret, _ = cap.read()
                if not ret:
                    break
        frames.append(np.array(batch_frames))

    # Match frames with captions
    data = list(zip(frames, captions[::batch_size * frame_step]))

    return data
