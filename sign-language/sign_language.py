import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load your trained CNN model (replace with your model path)
model = tf.keras.models.load_model("sign_language_mnist_model.h5")

# Load MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Labels for Sign Language MNIST
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def preprocess_image(frame, landmarks):
    """Preprocesses the hand image for Sign Language MNIST model."""
    if not landmarks:
        return None

    # Get bounding box of the hand
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    # Add padding to the bounding box
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_max = min(frame.shape[0], y_max + padding)

    # Crop the hand region
    hand_roi = frame[y_min:y_max, x_min:x_max]

    # Resize to 28x28 (Sign Language MNIST input size)
    hand_roi_resized = cv2.resize(hand_roi, (28, 28))

    # Convert to grayscale (Sign Language MNIST is grayscale)
    hand_roi_gray = cv2.cvtColor(hand_roi_resized, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values
    hand_roi_normalized = hand_roi_gray / 255.0

    # Reshape for model input (add channel dimension)
    hand_roi_input = np.expand_dims(hand_roi_normalized, axis=(0, 3))  # for channels_last format

    return hand_roi_input

# Initialize video capture
cap = cv2.VideoCapture(0)
text_output = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally and convert color space to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess the hand image for the model
            processed_image = preprocess_image(frame, hand_landmarks)

            if processed_image is not None:
                # Get model prediction
                prediction = model.predict(processed_image)
                predicted_label_index = np.argmax(prediction)
                predicted_label = labels[predicted_label_index]

                # Get confidence score (optional)
                confidence = np.max(prediction) * 100  # Convert to percentage

                # Always display the predicted label
                cv2.putText(image, f"Predicted: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Optional: Display confidence score
                # cv2.putText(image, f"Confidence: {confidence:.2f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Append the predicted label to the text output (optional)
                text_output += predicted_label + " "

    # Display the accumulated text output
    # cv2.putText(image, f"Output: {text_output}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the final image
    cv2.imshow('ASL Recognition', image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()