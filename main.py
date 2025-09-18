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

    # annotate fingertip pixel coordinates (x,y) based on current frame resolution
    tip_indices = [4, 8, 12, 16, 20]
    img_h, img_w = frame.shape[0], frame.shape[1]
    for idx in tip_indices:
        lm = hand_landmarks.landmark[idx]
        x_px, y_px = int(lm.x * img_w), int(lm.y * img_h)
        # offset text slightly so it does not overlap the landmark dot
        text_pos = (x_px + 6, y_px - 6)
        label = f"{x_px},{y_px}"
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 2, cv2.LINE_AA)


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


def compute_pinch(
    landmarks: List[Tuple[float, float, float]], image_width: int, image_height: int
) -> Tuple[bool, Tuple[int, int]]:
    """Return (is_pinch, (cx, cy)) using thumb tip (4) and index tip (8).

    A pinch is detected when the pixel distance between tips is below a
    dynamic threshold relative to current frame size.
    """
    if not landmarks:
        return False, (0, 0)

    def to_px(idx: int) -> Tuple[int, int]:
        lm = landmarks[idx]
        return int(lm[0] * image_width), int(lm[1] * image_height)

    thumb_x, thumb_y = to_px(4)
    index_x, index_y = to_px(8)
    dist = float(np.hypot(thumb_x - index_x, thumb_y - index_y))
    threshold = max(0.06 * min(image_width, image_height), 24.0)
    is_pinch = dist < threshold
    cx, cy = int((thumb_x + index_x) * 0.5), int((thumb_y + index_y) * 0.5)
    return is_pinch, (cx, cy)


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
    # draggable ball state
    ball_center: Tuple[int, int] | None = None
    ball_radius: int = 24
    dragging: bool = False
    drag_offset: Tuple[int, int] = (0, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # initialize ball in the center once we know the frame size
        if ball_center is None:
            h, w = frame.shape[0], frame.shape[1]
            ball_center = (w // 2, h // 2)

        total_raised = 0
        pinch_centers: List[Tuple[int, int]] = []
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                draw_landmarks(frame, hand_landmarks, None)
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                total_raised += count_raised_fingers(
                    lm_list, frame.shape[1], frame.shape[0]
                )
                is_pinch, pinch_xy = compute_pinch(
                    lm_list, frame.shape[1], frame.shape[0]
                )
                if is_pinch:
                    pinch_centers.append(pinch_xy)

        recent_counts.append(total_raised)
        display_count = Counter(recent_counts).most_common(1)[0][0] if recent_counts else 0

        # update draggable ball based on pinch
        if ball_center is not None:
            h, w = frame.shape[0], frame.shape[1]
            grab_radius = ball_radius + 32

            def clamp_to_bounds(x: int, y: int) -> Tuple[int, int]:
                return (
                    int(np.clip(x, ball_radius, w - ball_radius - 1)),
                    int(np.clip(y, ball_radius, h - ball_radius - 1)),
                )

            if dragging:
                if pinch_centers:
                    # follow the nearest pinch to the current ball center
                    px, py = min(
                        pinch_centers,
                        key=lambda p: (p[0] - ball_center[0]) ** 2 + (p[1] - ball_center[1]) ** 2,
                    )
                    new_x = px + drag_offset[0]
                    new_y = py + drag_offset[1]
                    ball_center = clamp_to_bounds(new_x, new_y)
                else:
                    dragging = False
            else:
                if pinch_centers:
                    # can start dragging only if pinching near the ball
                    px, py = min(
                        pinch_centers,
                        key=lambda p: (p[0] - ball_center[0]) ** 2 + (p[1] - ball_center[1]) ** 2,
                    )
                    dist2 = (px - ball_center[0]) ** 2 + (py - ball_center[1]) ** 2
                    if dist2 <= grab_radius * grab_radius:
                        dragging = True
                        drag_offset = (ball_center[0] - px, ball_center[1] - py)

            # draw the ball
            color = (0, 220, 80) if dragging else (0, 165, 255)
            cv2.circle(frame, ball_center, ball_radius, color, -1)
            cv2.circle(frame, ball_center, ball_radius, (30, 30, 30), 2)

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
