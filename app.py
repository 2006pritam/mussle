# app.py
import av
import cv2
import numpy as np
import mediapipe as mp
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class PushUpTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp_pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (50, 50, 50), -1)
        frame_bgr = cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0)

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

            angle = calculate_angle(shoulder, elbow, wrist)

            cv2.putText(frame_bgr, f'Elbow Angle: {int(angle)}°', (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)

            if angle > 160:
                self.stage = "up"
            if angle < 90 and self.stage == 'up':
                self.stage = "down"
                self.counter += 1

            cv2.putText(frame_bgr, f'Push-ups: {self.counter}', (width//2 - 150, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

            bar_fill = np.interp(angle, (90, 160), (350, 0))
            # draw progress bars left and right
            for i in range(int(350 - bar_fill), 350):
                if bar_fill != 0:
                    ratio = (i - (350 - bar_fill)) / bar_fill
                else:
                    ratio = 0
                color = (0, int(255 * ratio), 255 - int(255 * ratio))
                cv2.line(frame_bgr, (100, i + 150), (150, i + 150), color, 1)
                cv2.line(frame_bgr, (width - 150, i + 150), (width - 100, i + 150), color, 1)

            cv2.rectangle(frame_bgr, (100, 150), (150, 500), (255, 255, 255), 2)
            cv2.rectangle(frame_bgr, (width - 150, 150), (width - 100, 500), (255, 255, 255), 2)

            cv2.circle(frame_bgr, (125, 150), 10, (255, 0, 255), -1)
            cv2.circle(frame_bgr, (width - 125, 150), 10, (255, 0, 255), -1)

            mp_drawing.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
            )

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

# Streamlit UI
st.title("Push-Up Tracker — Streamlit + MediaPipe")
st.write("Press **Start** to allow camera access; perform push-ups in view of your webcam. (Left arm used for angle calc by default.)")

ctx = webrtc_streamer(
    key="pushup",
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    ),
    video_transformer_factory=PushUpTransformer,
    async_transform=True,
)

if ctx.state.playing:
    st.success("Camera is active — performing detection.")
else:
    st.info("Camera stopped.")
