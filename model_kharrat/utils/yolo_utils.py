import cv2
import numpy as np

def load_yolo_model(weights_path, config_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def detect_cars(frame, net, classes, conf_threshold=0.5, nms_threshold=0.4):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] == 'car':
                box = detection[:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result_boxes = [boxes[i[0]] for i in indices]

    return result_boxes

def calculate_centroid(box):
    x, y, w, h = box
    centroid_x = x + w // 2
    centroid_y = y + h // 2
    return centroid_x, centroid_y
