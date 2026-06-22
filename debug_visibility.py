import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose

video_path = None
for root, dirs, files in os.walk("bicep_curl"):
    for f in files:
        if f.endswith((".mov", ".mp4")) and not f.startswith("."):
            video_path = os.path.join(root, f)
            break
    if video_path:
        break

print(f"Testing: {video_path}")
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    checked = 0
    detected = 0
    for _ in range(300): 
        ret, frame = cap.read()
        if not ret:
            break
        checked += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            detected += 1
            if detected <= 3:  
                lms = results.pose_landmarks.landmark
                S = mp_pose.PoseLandmark
                for name in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
                              "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
                              "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]:
                    lm = lms[getattr(S, name).value]
                    print(f"  {name}: visibility={lm.visibility:.3f}")
                print()

cap.release()
print(f"\nFrames checked: {checked}, poses detected: {detected} ({100*detected//max(checked,1)}%)")
