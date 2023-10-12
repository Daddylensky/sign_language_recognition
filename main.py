import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model('sign_language_model.h5')
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape


while True:
    _, frame = cap.read()
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
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
            frame = mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

            # Preprocess the image: resize and normalize
            input_frame = cv2.resize(frame, (224, 224))
            input_frame = np.expand_dims(input_frame, axis=0)
            input_frame = input_frame / 255.0

            # Make prediction
            prediction = model.predict(input_frame)
            predicted_class = np.argmax(prediction[0])

            # Map the predicted class index to the corresponding letter
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']  # Replace with your actual labels
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
