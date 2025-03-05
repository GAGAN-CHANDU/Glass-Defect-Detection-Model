import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tempfile
import pyttsx3  # For text-to-speech

# Initialize Roboflow and get your model
api_key = "IC8YcLgRE0ll7gW5sFGo"  # Replace with your Roboflow API key
workspace = "research-azmux"  # Replace with your workspace
project_name = "deep-learning-project-t1ydm"  # Replace with your project name
version = "4"  # Replace with your model version

# Set up the Roboflow API
rf = Roboflow(api_key=api_key)
model = rf.workspace(workspace).project(project_name).version(version).model

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Streamlit App Title and Description
st.set_page_config(page_title="Defect Detection", layout="wide", page_icon="🔍")
st.title("🔍 Glass Defect Detection Model")
st.markdown("""
### Welcome to the **Glass Defect Detection App**!  
Powered by **Roboflow** and **YOLOv8**, this app helps you identify potential defects in glass.  

#### How it works:
- **Upload an Image**: Choose an image file from your device.  
- **Capture an Image**: Use your camera to take a photo directly within the app.  
- **Analyze the Image**: The model will process the image to detect defects.  
- **View Results**: Detected defects are highlighted with bounding boxes and class labels.  

Interactively explore the results and see detailed predictions for each detection!
""")

# Sidebar for instructions and additional settings
st.sidebar.title("📋 Instructions")
st.sidebar.info("""
1. Upload an image or capture one using the camera.  
2. Adjust confidence threshold if necessary.  
3. View results with bounding boxes and class labels.
""")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 40, 5)
overlap_threshold = st.sidebar.slider("Overlap Threshold (%)", 0, 100, 30, 5)

# Image input options
st.subheader("Select Image Source")
image_source = st.radio("Choose an image input method:", ("Upload Image", "Capture from Camera"))

# Handle image input
if image_source == "Upload Image":
    uploaded_image = st.file_uploader("📁 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    image = Image.open(uploaded_image).convert("RGB") if uploaded_image else None
elif image_source == "Capture from Camera":
    camera_image = st.camera_input("📸 Capture an image using your camera")
    image = Image.open(camera_image).convert("RGB") if camera_image else None

if image:
    # Display the input image
    st.subheader("Input Image")
    col1, col2 = st.columns(2)
    with col1:
        # st.image(image, caption="Original Image", use_column_width=True)
        st.image(image, caption="Original Image", use_container_width=True)

    # Save the input image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format="JPEG")
        temp_image_path = temp_file.name

    # Run inference
    st.subheader("🔄 Running inference... Please wait.")
    try:
        # Send the image path to Roboflow for inference
        response = model.predict(temp_image_path, confidence=confidence_threshold, overlap=overlap_threshold).json()

        # Prepare the output image with bounding boxes
        if response.get('predictions'):
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf", 20)  # Use a system font
            colors = {"Defected": "red", "None": "blue"}  # Color mapping for classes

            has_defect = False  # Flag to track if any defect is detected

            for prediction in response['predictions']:
                if prediction['class'] == "None":
                    continue
                x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
                confidence = prediction['confidence']
                obj_class = prediction['class']

                # Check for defected class
                if obj_class == "Defected":
                    has_defect = True

                # Calculate bounding box coordinates
                left = x - width / 2
                top = y - height / 2
                right = x + width / 2
                bottom = y + height / 2

                # Determine the color for the class
                color = colors.get(obj_class, "green")

                # Draw the bounding box
                draw.rectangle([left, top, right, bottom], outline=color, width=3)

                # Prepare and draw the label text
                text = f"{obj_class} ({confidence*100:.2f}%)"
                text_size = draw.textbbox((0, 0), text, font=font)  # Calculate text size
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]

                # Draw the label background
                draw.rectangle(
                    [left, top, left + text_width + 4, top + text_height + 4],
                    fill=color,
                )
                draw.text((left + 2, top + 2), text, fill="white", font=font)

            # Play appropriate voice message
            if has_defect:
                st.warning("Defects in Glass!")
                engine.say("Defects in Glass")
            else:
                st.success("No defects in glass.")
                engine.say("No defects in glass")

            engine.runAndWait()

            # Display the annotated image
            with col2:
                # st.image(image, caption="Image with Bounding Boxes", use_column_width=True)
                st.image(image, caption="Image with Bounding Boxes", use_container_width=True)

            # Display prediction details in an expandable section
            with st.expander("📊 Prediction Details"):
                for i, prediction in enumerate(response['predictions'], 1):
                    if prediction['class'] == "None":
                        continue
                    st.markdown(f"""
                    **Detection {i}:**  
                    - **Class:** {prediction['class']}  
                    - **Confidence:** {prediction['confidence']*100:.2f}%  
                    - **Bounding Box:** (x: {prediction['x']}, y: {prediction['y']}, width: {prediction['width']}, height: {prediction['height']})  
                    """)
        else:
            st.warning("No defects detected in glass.")
            engine.say("No defects in glass")
            engine.runAndWait()
    except Exception as e:
        st.error(f"An error occurred during inference: {e}")

# Footer with additional info
# st.sidebar.markdown("---")
# st.sidebar.caption("Made with ❤️ using Streamlit and Roboflow.")
