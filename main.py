"""
hand pose tracker using mediapipe and opencv

controls:
  q quit
  s save snapshot

"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import List, Tuple

import cv2
import numpy as np

try:
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe import solutions as mp_solutions
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "MediaPipe is required. Install with: pip install mediapipe==0.10.14"
    ) from exc


def draw_landmarks(frame: np.ndarray, hand_landmarks, handedness: str | None) -> None:
    mp_drawing = mp_solutions.drawing_utils
    mp_styles = mp_solutions.drawing_styles
    mp_hands = mp_solutions.hands

    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )

    if handedness:
        h_label = f"{handedness} hand"
        h_wrist = hand_landmarks.landmark[0]
        h_x, h_y = int(h_wrist.x * frame.shape[1]), int(h_wrist.y * frame.shape[0])
        cv2.putText(frame, h_label, (h_x + 10, h_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def count_raised_fingers(landmarks: List[Tuple[float, float, float]], image_width: int, image_height: int) -> int:
    if not landmarks:
        return 0

    def to_px(idx: int) -> Tuple[int, int]:
        lm = landmarks[idx]
        return int(lm[0] * image_width), int(lm[1] * image_height)

    # heuristic: for each finger, compare tip and pip along y for open/closed
    # for thumb, compare x relative to wrist depending on left/right
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    wrist_x, wrist_y = to_px(0)

    raised = 0
    # thumb
    thumb_tip_x, thumb_tip_y = to_px(4)
    thumb_ip_x, thumb_ip_y = to_px(3)
    if abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x):
        raised += 1

    # other fingers (index to pinky)
    for tip_idx, pip_idx in zip(tips[1:], pips[1:]):
        tip_x, tip_y = to_px(tip_idx)
        pip_x, pip_y = to_px(pip_idx)
        if tip_y < pip_y - 10:  # open if tip is above pip in image coords
            raised += 1
    return int(np.clip(raised, 0, 5))


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try a different camera index.")

    mp_hands = mp_solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print("Press 'q' to quit, 's' to save a snapshot.")
    recent_counts = deque(maxlen=7)
    last_saved = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        total_raised = 0
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                draw_landmarks(frame, hand_landmarks, None)
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                total_raised += count_raised_fingers(
                    lm_list, frame.shape[1], frame.shape[0]
                )

        recent_counts.append(total_raised)
        display_count = Counter(recent_counts).most_common(1)[0][0] if recent_counts else 0

        cv2.putText(frame, f"Raised Fingers: {display_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 255, 50), 3, cv2.LINE_AA)
        cv2.imshow("Hand Pose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            now = time.time()
            if now - last_saved > 0.5:
                fname = f"hand_snapshot_{int(now)}.png"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                last_saved = now

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
