from time import time
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import numpy as np


def get_faces_landmarks(img: np.array):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    faces = []

    results_object = face_mesh.process(img_rgb)
    results = results_object.multi_face_landmarks
    if results:
        for landmark in results:
            # mp_draw.draw_landmarks(img, landmark, face.FACE_CONNECTIONS)
            face_data = []
            for id, lm in enumerate(landmark.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                face_data.append((x, y))
            faces.append(face_data)

    return faces


def get_face_bbox(face_points: list) -> tuple:
    region = np.array(face_points)

    min_x = np.min(region[:, 0])
    min_y = np.min(region[:, 1])
    max_x = np.max(region[:, 0])
    max_y = np.max(region[:, 1])

    return min_x, min_y, max_x, max_y


classes = {
    0: {"ClassName": "Mask", "Color": (0, 200, 0)},
    1: {"ClassName": "NoMask", "Color": (0, 0, 200)},
}

model = load_model("MaskModel2.h5")
face = mp.solutions.face_mesh
face_mesh = face.FaceMesh(max_num_faces=3, min_detection_confidence=.4, min_tracking_confidence=.4)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

p_time = 0
while cap.isOpened():
    success, img = cap.read()

    img = cv2.flip(img, 1)
    faces = get_faces_landmarks(img)

    if faces:
        face_imgs = []
        face_bboxs = []

        for face in faces:
            x1, y1, x2, y2 = get_face_bbox(face)

            face_cropped = img[y1:y2, x1:x2]
            face_cropped = cv2.resize(face_cropped, (224, 224))
            face_imgs.append(face_cropped)
            face_bboxs.append((x1, y1, x2, y2))

        if face_imgs:
            data = np.array(face_imgs)
            predictions = model.predict(data)
            for pred, bbox in zip(predictions, face_bboxs):
                x1, y1, x2, y2 = bbox
                class_id = np.argmax(pred)
                confidence = pred[class_id]

                class_name = classes[class_id]['ClassName']
                color = classes[class_id]['Color']

                cv2.putText(img, f"{class_name} {int(round(confidence, 2) * 100)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 200), 2)
    cv2.imshow("Res", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()
