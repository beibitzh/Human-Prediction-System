import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load trained model
model = tf.keras.models.load_model("action_model.h5")

# MUST match label_encoder.classes_ order:
# 0 calling
# 1 clapping
# 2 cycling
# 3 dancing
# 4 drinking
# 5 eating
# 6 fighting
# 7 hugging
# 8 laughing
# 9 listening_to_music
# 10 running
# 11 sitting
# 12 sleeping
# 13 texting
# 14 using_laptop
classNames = [
    "calling",
    "clapping",
    "cycling",
    "dancing",
    "drinking",
    "eating",
    "fighting",
    "hugging",
    "laughing",
    "listening_to_music",
    "running",
    "sitting",
    "sleeping",
    "texting",
    "using_laptop",
]

inputSize = (224, 224)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: cannot open webcam")
    exit()

print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV gives BGR -> convert to RGB (what ResNet expects)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(frame_rgb, inputSize)
    img = img.astype(np.float32)

    # Same preprocessing as training
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x, verbose=0)
    scores = preds[0]
    classId = int(np.argmax(scores))
    prob = float(np.max(scores))

    labelText = f"{classNames[classId]} ({prob * 100:.1f}%)"

    # Draw label on original BGR frame
    cv2.putText(
        frame,
        labelText,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Webcam Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
