import cv2
import mediapipe as mp
from deepface import DeepFace
import time
from collections import Counter

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

def draw_emotion_counts(img, emotion_counts):
    h, w, _ = img.shape
    x0, y0 = 10, h - 30
    dy = 30
    for i, (emotion, count) in enumerate(emotion_counts.items()):
        y = y0 - i * dy
        # cv2.putText(img, f'{emotion}: {count}', (x0, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

def print_emotion_summary(emotion_counts):
    total_emotions = sum(emotion_counts.values())
    print("\nEmotion Summary:")
    for emotion, count in emotion_counts.items():
        accuracy = (count / total_emotions) * 100 if total_emotions > 0 else 0
        print(f'{emotion}: {count} ({accuracy:.2f}%)')
    print(f'Total emotions detected: {total_emotions}')


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # Set resolution width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set resolution height

    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
    emotion_counts = Counter()

    while True:
        success, img = cap.read()
        if not success:
            continue

        img, faces = detector.findFaceMesh(img)
        if faces:
            try:
                analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                if analysis and isinstance(analysis, list):
                    emotions = analysis[0]['emotion']
                    dominant_emotion = max(emotions, key=emotions.get)  # Get the dominant emotion
                    emotion_counts[dominant_emotion] += 1
                    draw_emotion_counts(img, emotion_counts)
                    print(f'Dominant Emotion: {dominant_emotion}')
            except Exception as e:
                print(f'Error in emotion analysis: {e}')

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print_emotion_summary(emotion_counts)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
