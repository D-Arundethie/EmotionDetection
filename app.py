import cv2
import mediapipe as mp
from deepface import DeepFace
from flask import Flask, request, jsonify
import numpy as np
from collections import Counter

app = Flask(__name__)

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

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    detector = FaceMeshDetector(maxFaces=1)
    img, faces = detector.findFaceMesh(img)

    if faces:
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            if analysis and isinstance(analysis, list):
                emotions = analysis[0]['emotion']
                dominant_emotion = max(emotions, key=emotions.get)
                return jsonify({'emotion': dominant_emotion})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'No face detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)
