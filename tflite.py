# tflite.py
# Run in tflite_env (TF 2.13):

import os
import sys
import numpy as np

KERAS_MODEL    = "v1exhaustiveroiasl_landmark_model.keras"
WEIGHTS_FILE   = "asl_weights_tmp.npz"
TFLITE_OUT     = "v1exhaustiveroiasl_landmark_model.tflite"
IMG_SIZE       = 200
NUM_CLASSES    = 26
PROJECT_PYTHON = "/Users/steverinoma/PycharmProjects/mutual_intelligence/venv/bin/python3"

# ── Step 1: extract weights from .keras using project venv (TF 2.16) ─────────
if not os.path.exists(WEIGHTS_FILE):
    print(f"Step 1: extracting weights from {KERAS_MODEL}...")
    import subprocess
    result = subprocess.run([
        PROJECT_PYTHON, "-c",
        f"""
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('{KERAS_MODEL}')
weights = [w.numpy() for w in model.weights]
np.savez('{WEIGHTS_FILE}', *weights)
print(f"Saved {{len(weights)}} weight arrays")
for i, w in enumerate(weights):
    print(f"  w{{i}}: {{w.shape}}")
"""
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR extracting weights:")
        print(result.stderr)
        sys.exit(1)
else:
    print(f"Step 1: {WEIGHTS_FILE} already exists, skipping.")

# ── Step 2: rebuild architecture in TF 2.13 and load weights ─────────────────
import tensorflow as tf
print(f"\nStep 2: rebuilding model in TF {tf.__version__}...")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same',
                           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
])

# Load weights
print("Loading weights...")
data    = np.load(WEIGHTS_FILE)
weights = [data[f"arr_{i}"] for i in range(len(data.files))]
model.set_weights(weights)
print(f"Loaded {len(weights)} weight arrays ✓")

# Quick sanity check
dummy  = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
result = model(dummy, training=False)
print(f"Forward pass OK: {result.shape} ✓")

# ── Step 3: convert to TFLite ─────────────────────────────────────────────────
print(f"\nStep 3: converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_bytes)
print(f"Saved: {TFLITE_OUT} ({os.path.getsize(TFLITE_OUT)/1024:.1f} KB)")

# ── Verify ────────────────────────────────────────────────────────────────────
print("\nVerifying...")
interp = tf.lite.Interpreter(model_path=TFLITE_OUT)
interp.allocate_tensors()
inp_d = interp.get_input_details()
out_d = interp.get_output_details()
print(f"  Input  : {inp_d[0]['shape']}  {inp_d[0]['dtype']}")
print(f"  Output : {out_d[0]['shape']}  {out_d[0]['dtype']}")

interp.set_tensor(inp_d[0]['index'], dummy)
interp.invoke()
out = interp.get_tensor(out_d[0]['index'])
print(f"  Inference OK: {out.shape}  ✓")

if os.path.exists("class_names.txt"):
    with open("class_names.txt") as f:
        classes = [l.strip() for l in f.readlines()]
    match = out.shape[1] == len(classes)
    print(f"  Classes: {len(classes)} {'✓' if match else 'WARNING: mismatch'}")

# Clean up temp weights file
os.remove(WEIGHTS_FILE)
print(f"\nDone. Copy {TFLITE_OUT} and class_names.txt to ~/asl/ on the Pi.")
