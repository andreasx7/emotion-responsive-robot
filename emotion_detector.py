import cv2
from fer import FER

# Initialize webcam and detector
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect emotion
    emotion, score = detector.top_emotion(frame)

    if emotion:
        label = f"{emotion.upper()} ({score:.2f})"
        print("Detected:", label)
        # Show on frame
        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("FER Webcam Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
