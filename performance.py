import cv2
import mediapipe as mp
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import numpy as np
import time


class FaceMeshDetector:
    def __init__(self, StaticMode=False, maxFaces=2, minDetectionCon=0.8, minTrackCon=0.8):
        self.StaticMode = StaticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.StaticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLandmarks in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks,
                                               self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)

                face = []
                for lm in faceLandmarks.landmark:
                    ih, iw, _ = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set resolution height

    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
    emotion_counts = Counter()
    true_labels = []  # Ground truth labels
    predicted_labels = []  # Model predictions

    while True:
        success, img = cap.read()
        if not success:
            continue

        img, faces = detector.findFaceMesh(img)
        if faces:
            try:
                analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                if analysis and isinstance(analysis, list) and 'emotion' in analysis[0]:
                    emotions = analysis[0]['emotion']
                    dominant_emotion = max(emotions, key=emotions.get)  # Get the dominant emotion
                    emotion_counts[dominant_emotion] += 1
                    predicted_labels.append(dominant_emotion)

                    # Add your ground truth labels here
                    true_labels.append('happy')  # Replace with actual ground truth label for the frame

                    print(f'Dominant Emotion: {dominant_emotion}')
            except Exception as e:
                print(f'Error in emotion analysis: {e}')

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Evaluate model performance
    evaluate_model(true_labels, predicted_labels)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
