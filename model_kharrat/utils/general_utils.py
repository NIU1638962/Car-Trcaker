import cv2

def load_video(path, video_name):
    return cv2.VideoCapture(f"{path}/{video_name}")

def read_frame(video, fps):
    video.set(cv2.CAP_PROP_FPS, fps)
    ret, frame = video.read()
    if not ret:
        return None
    return frame
