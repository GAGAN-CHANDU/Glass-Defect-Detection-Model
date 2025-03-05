# Glass Defect Detection App

## 📌 Overview
This Streamlit-based web application detects defects in glass using a pre-trained **YOLOv8** model hosted on **Roboflow**. It allows users to upload or capture images, analyzes them for defects, and provides real-time feedback with bounding boxes.

## 🚀 Features
- **Upload Image:** Choose an image file (JPG, JPEG, PNG) from your device.
- **Capture Image:** Use your camera to take a picture directly within the app.
- **Run Inference:** Detect defects using a trained model.
- **Bounding Box Visualization:** Highlight detected defects with bounding boxes and labels.
- **Confidence & Overlap Threshold Adjustment:** Modify detection settings from the sidebar.
- **Text-to-Speech Support:** Announces whether defects are found or not.
- **Prediction Details:** View detection confidence and bounding box dimensions.

## 🛠️ Technologies Used
- **Python**
- **Streamlit** (for UI)
- **Roboflow** (for YOLOv8 inference)
- **Pillow (PIL)** (for image processing)
- **pyttsx3** (for text-to-speech)
- **Tempfile** (for handling temporary image storage)

## 🔧 Setup & Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/glass-defect-detection.git
   cd glass-defect-detection
   ```
2. Install dependencies:
   ```sh
   pip install streamlit pillow roboflow pyttsx3
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## 🎯 Usage
1. Launch the app and select an image input method.
2. Adjust confidence and overlap thresholds if needed.
3. Click **Analyze Image** to detect defects.
4. View the processed image with bounding boxes and prediction details.


---
💡 Developed with ❤️ using **Streamlit** and **YOLOv8**

