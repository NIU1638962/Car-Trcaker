import os
import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path):
    model = YOLO(model_path)  # Carga el modelo directamente desde ultralytics
    return model

def detect_cars(model, frame):
    results = model(frame)  # Detectar coches en el frame
    detections = results[0]  # Obtener detecciones
    boxes = []

    for *box, conf, cls in detections:  
        if int(cls) == 2:  
            boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), conf.item()])

    return boxes

def calculate_centroid(box):
    x1, y1, x2, y2, conf = box
    centroid_x = int((x1 + x2) / 2)
    centroid_y = int((y1 + y2) / 2)
    return centroid_x, centroid_y

def track_vehicles(centroids, previous_centroids, max_distance=50):
    matches = []
    for c in centroids:
        distances = [np.linalg.norm(np.array(c) - np.array(pc)) for pc in previous_centroids]
        if distances:  
            min_distance = min(distances)
            if min_distance < max_distance:
                matched_idx = distances.index(min_distance)
                matches.append((c, previous_centroids[matched_idx]))
    return matches

def detect_entry_exit(centroid, entry_line_y, exit_line_y):
    _, y = centroid
    if y < entry_line_y:
        return 'enter'
    elif y > exit_line_y:
        return 'exit'
    return 'none'

def load_video(video_path):
    return cv2.VideoCapture(video_path)

def read_frame(video, frame_number):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    return frame if ret else None

def main():
   
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    
    model_path = os.path.join(current_directory, "yolov5.pt")  
    model = load_yolo_model(model_path)

    # Cargar el video
    video_path = os.path.join(current_directory, "video.mp4")  
    video = load_video(video_path)
    previous_centroids = []
    
    while True:
        frame = read_frame(video, 30)  
        if frame is None:
            break

        # Detectar coches en el frame
        boxes = detect_cars(model, frame)
        centroids = [calculate_centroid(box) for box in boxes]

        # Hacer seguimiento
        matches = track_vehicles(centroids, previous_centroids)

        # Detectar entrada o salida
        for centroid in centroids:
            status = detect_entry_exit(centroid, entry_line_y=200, exit_line_y=500)
            print(f'Vehicle status: {status}')

        # Guardar centroides actuales para el próximo frame
        previous_centroids = centroids

    video.release()

if __name__ == "__main__":
    main()
