# Mutual - ASL Live-Fingerspelling Classification and LLM Processing
**Mutual**, a student non-profit centered in Brookline, Massachusetts, is dedicated to bridging communication between the Deaf community and non-signers. This repository features our ASl fingerspelling hand pose recognition pipeline: a hand-pose landmark classification model paired with a large-language-model processing layer that turns raw letter predictions into fluent, readable text. 
Our most recent updates and packages, including **piroitranslation.py**, **asl_settings.py**, **v2piroiasl_landmark_model.keras** are compatible on the Raspberry Pi OS ecosystem as we develop a wearable designed for live ASl fingerspelling translation. This pipeline combined Mediapipe for hand detection and 21-point landmark extraction, Tensorflow/Keras for trained classification model, and an LLM post-processing stage that forms grammatical sentences from strings of detected hand signs. Our program runs on Raspberry Pi, translating fingerspelling input into spoken and on-screen English in real time. 



