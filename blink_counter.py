import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

blink_count = 0
eye_closed = False


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def eye_aspect_ratio(landmarks, eye):
    vertical_1 = distance(landmarks[eye[1]], landmarks[eye[5]])
    vertical_2 = distance(landmarks[eye[2]], landmarks[eye[4]])
    horizontal = distance(landmarks[eye[0]], landmarks[eye[3]])

    return (vertical_1 + vertical_2) / (2 * horizontal)


while True:
    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ratio = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ratio = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ratio + right_ratio) / 2

        if ear < 0.20:
            eye_closed = True
        else:
            if eye_closed:
                blink_count += 1
                eye_closed = False

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Blink Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
