import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model('sign_language_model.h5')
print("here\n\n")
print(model.output_shape)
print("\n\n")
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    h, w, c = frame.shape  # Add this line to define frame dimensions
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x, x_max)
                x_min = min(x, x_min)
                y_max = max(y, y_max)
                y_min = min(y, y_min)

            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)

            roi = frame[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:  # Skip if the roi is empty
                continue

            input_roi = cv2.resize(roi, (224, 224))
            input_roi = np.expand_dims(input_roi, axis=0)
            input_roi = input_roi / 255.0

            # Make prediction
            prediction = model.predict(input_roi)
            predicted_class = np.argmax(prediction[0])
            
            # Map the predicted class index to the corresponding letter
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
            predicted_label = labels[predicted_class]

            # Display the prediction on the frame
            cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
