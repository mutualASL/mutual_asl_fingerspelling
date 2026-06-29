# Mutual - ASL Live-Fingerspelling Classification and LLM Processing
**Mutual**, a student non-profit centered in Brookline, Massachusetts, is dedicated to bridging communication between the Deaf community and non-signers. This repository features our ASl fingerspelling hand pose recognition pipeline: a hand-pose landmark classification model paired with a large-language-model processing layer that turns raw letter predictions into fluent, readable text. 

Our most recent updates and packages, including **piroitranslation.py**, **asl_settings.py**, **v2piroiasl_landmark_model.keras** are compatible on the Raspberry Pi OS ecosystem as we develop a wearable designed for live ASl fingerspelling interpretation. This pipeline combined Mediapipe for hand detection and 21-point landmark extraction, Tensorflow/Keras for a trained classification model, and an LLM post-processing stage that forms grammatical sentences from strings of detected hand signs. Our program runs on Raspberry Pi, translating fingerspelling input into spoken and on-screen English in real time. 

## Methodology
### 1. Data Collection and Preprocessing:
1a. We built our own image collector that captures fingerspelling hand poses as rendered skeleton images. Each capture features Mediapipe's 21 hand landmarks and fits them into a dynamic bounding box that motion tracks the hand. The samples are collected with auto rotation adjustments so the hand is otherwise upright or horizontal. Landmark connections are drawn on a 200x200 black-and-white canvas. The design of the pipeline attempts to mitigate lighting and background information by concentrating only on the Mediapipe hand landmark skeleton. These ASL hand pose images are collected into separate folders, each corresponding to an English alphabet letter. 

<table align="center">
  <tr>
    <td align="center">
      <img height="300" alt="Data collection running on Mac webcam" src="https://github.com/user-attachments/assets/f57d14d0-2341-4498-b762-b2332ab52847" />
    </td>
    <td align="center">
      <img height="300" alt="Data folders" src="https://github.com/user-attachments/assets/5e69e6cc-6705-4a6e-b411-49bdee4ead70" />
    </td>
  </tr>
  <tr>
    <td align="center"><sub>Data collection running on Mac webcam with live hand motion tracking and MediaPipe processing</sub></td>
    <td align="center"><sub>Data folders</sub></td>
  </tr>
</table>

1b. With the integrated auto rotation adjustment logic that "standardizes" hand positions, camera-hand tracking, dynamic bounding box defined by hand landmarks, and skeleton rendering, the model can focus on learning handshape and tackling angle variations. These processing functions are also applied during live translation, allowing our compact VGG-style CNN to better handle unpredictable hand angles and positions. 

### 2. Model Training: 
2. Our model training has undergone many iterations. Currently, the classifier is a convolutional neural network that takes the 200x200 black-and-white skeleton canvas and outputs a probability over the 26 letters of the English alphabet. The body is three convolutional blocks — each one Conv2D → BatchNorm → MaxPool → Dropout, doubling the filters as it goes (16 → 32 → 64) — followed by a flatten, a dense layer, and the 26-way softmax output. BatchNorm and dropout throughout keep it from overfitting the training captures.

About 2.59M trainable parameters. The saved .keras weighs ~30 MB because it stores optimizer state; the weights themselves are ~10 MB, and we strip the rest for Pi deployment.

```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 200, 200, 16)   │           160 │
│ batch_normalization (BatchNorm) │ (None, 200, 200, 16)   │            64 │
│ max_pooling2d (MaxPooling2D)    │ (None, 100, 100, 16)   │             0 │
│ dropout (Dropout)               │ (None, 100, 100, 16)   │             0 │
│ conv2d_1 (Conv2D)               │ (None, 100, 100, 32)   │         4,640 │
│ batch_normalization_1           │ (None, 100, 100, 32)   │           128 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 50, 50, 32)     │             0 │
│ dropout_1 (Dropout)             │ (None, 50, 50, 32)     │             0 │
│ conv2d_2 (Conv2D)               │ (None, 50, 50, 64)     │        18,496 │
│ batch_normalization_2           │ (None, 50, 50, 64)     │           256 │
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 25, 25, 64)     │             0 │
│ dropout_2 (Dropout)             │ (None, 25, 25, 64)     │             0 │
│ flatten (Flatten)               │ (None, 40000)          │             0 │
│ dense (Dense)                   │ (None, 64)             │     2,560,064 │
│ batch_normalization_3           │ (None, 64)             │           256 │
│ dropout_3 (Dropout)             │ (None, 64)             │             0 │
│ dense_1 (Dense)                 │ (None, 26)             │         1,690 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params:        2,585,754
 Trainable params:    2,585,402
 Non-trainable params:      352
```

