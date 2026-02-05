import cv2
from fer import FER

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, "AI FACE EMOTION HUD", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

    result = detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        dominant = max(emotions, key=emotions.get)
        score = emotions[dominant]

        # Face box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Dominant emotion
        cv2.putText(frame, f"{dominant} ({int(score*100)}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

        # ---- LEFT SIDE EMOTION BARS ----
        y0 = 60
        for emo, val in emotions.items():
            bar_length = int(val * 150)

            cv2.putText(frame, emo, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.rectangle(frame, (70, y0 - 10),
                          (70 + bar_length, y0),
                          (255, 0, 255), -1)

            y0 += 25

    cv2.imshow("AI FACE EMOTION HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

