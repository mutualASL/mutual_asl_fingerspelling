# Mutual - ASL Live-Fingerspelling Classification and LLM Processing
**Mutual**, a student non-profit centered in Brookline, Massachusetts, is dedicated to bridging communication between the Deaf community and non-signers. This repository features our ASl fingerspelling hand pose recognition pipeline: a hand-pose landmark classification model paired with a large-language-model processing layer that turns raw letter predictions into fluent, readable text. 

Our most recent updates and packages, including **piroitranslation.py**, **asl_settings.py**, **v2piroiasl_landmark_model.keras** are compatible on the Raspberry Pi OS ecosystem as we develop a wearable designed for live ASl fingerspelling interpretation. This pipeline combined Mediapipe for hand detection and 21-point landmark extraction, Tensorflow/Keras for a trained classification model, and an LLM post-processing stage that forms grammatical sentences from strings of detected hand signs. Our program runs on Raspberry Pi, translating fingerspelling input into spoken and on-screen English in real time. 

## Methodology
### 1. Data collection:

We built our own image collector that captures fingerspelling hand poses as rendered skeleton images. Each capture features Mediapipe's 21 hand landmarks and fits them into a dynamic bounding box that motion tracks the hand. The samples are collected with auto rotation-adjustments so the hand is otherwise upright or horizontal. Landmark connections are drawn on a 200x200 black-and-white canvas. The design of the pipeline attempts to mitigate lighting and background information by concentrating only on the Mediapipe hand landmarks skeleton. These ASL hand pose images are collected into separate folders, each corresponding to an English alphabet letter. 

<img width="497" height="382" alt="Screenshot 2026-06-29 at 3 38 17 PM" src="https://github.com/user-attachments/assets/f57d14d0-2341-4498-b762-b2332ab52847" /> <img width="1124" height="458" alt="Screenshot 2026-06-30 at 12 04 29 AM" src="https://github.com/user-attachments/assets/5e69e6cc-6705-4a6e-b411-49bdee4ead70" />





