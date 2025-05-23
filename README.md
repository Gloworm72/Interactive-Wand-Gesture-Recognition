# Interactive Wand

A personal passion project recreating the magic of spellcasting through computer vision, machine learning, and themed show control — all powered by a Raspberry Pi 5 and written entirely in Python.

---

## Project Summary

This wand system detects spellcasting gestures in real-time using OpenCV and an infrared-lit wand. It recognizes and responds to two specific spells:

- **"Alohamora"** — opens the magical box with warm purple fire  
- **"Colloportus"** — closes it with a cool burst of blue flame  

The system features:

- Real-time IR blob tracking and wand path tracing
- Spell recognition using a trained SVM classifier
- Servo-based box movement
- Custom LED animations tied to spell type
- Themed sound effects with seamless background music
- Filtering to prevent false or accidental spell detection

All code runs on-device using multithreaded Python and a Pi Camera.

---

## Technologies Used

- `OpenCV` for video input and motion tracking  
- `scikit-learn` SVM with `GridSearchCV` for spell classification  
- `Pi5Neo` to control RGB LED strip over SPI  
- `pygame` for real-time sound effects and music  
- `pigpio` and `gpiozero` for hardware PWM and servo control  
- Custom wand trace dataset of 400+ samples, labeled and trained manually  
- Threading to keep vision, servo, LED, and audio systems responsive  

---

## Spellcasting Flow

![Wand (1)](https://github.com/user-attachments/assets/949b9146-4611-4c83-a0c0-e3fd67cafff5)

---

## File Overview

**HarryPotterWandcv.py**

↳ Main runtime script: blob detection, trace drawing, spell prediction, and show control.

**HarryPotterWandsklearn.py**

↳ Used to run the pre-trained SVM classifier concurrently.

**new_custom_classifier.pkl**

↳ Pre-trained model for classifying spells based on trace shape.

**lastframe.jpg**

↳ Latest wand trace visualization, saved for debugging or training.

**Sounds/**

↳ Sound effects and background music used in spellcasting.

**DatasetCreation/**

↳ Python for drawing custom training data, converting that training data into the correct format, training the SVM classifier to produce the .pkl file

---

## ML & Classification

I created a custom dataset by collecting over 400 wand path traces drawn in-air. These were:

- Centered and normalized
- Smoothed and resampled
- Converted to vector features

I used `GridSearchCV` to tune a Support Vector Machine (SVM) classifier that could distinguish between gestures with over 99% accuracy.

The classifier runs on-device in real time with minimal latency.

---

## Show Control Highlights

- **Servo Logic** – Smooth actuation of box lid using hardware PWM and `pigpio`  
- **LED FX** – Custom “fire” animations with randomized color flickers using `Pi5Neo`  
- **Audio Layers** – Spell SFX mixed over looping background music via `pygame`  
- **Gesture Filtering** – Start and stop conditions prevent noisy traces from triggering spells  

---

## 🎥 Demo Video

[![Watch the video](https://img.youtube.com/vi/IFpQFHPK7W4/0.jpg)](https://www.youtube.com/watch?v=IFpQFHPK7W4)

*Click the image to watch the full demo.*

---

## Final Thoughts

This was one of the most technically rewarding projects I've created — combining embedded hardware, computer vision, machine learning, and interactive storytelling. It’s a small glimpse into how software and show control can bring magic to life.
