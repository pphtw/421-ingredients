
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input

# รายชื่อคลาส (ปรับตามชุดข้อมูลของคุณ)
class_names = ['avocado', 'beetroot', 'cabbage', 'carrots', 'plum tomatoes']

# ฟังก์ชันประมวลผลและทำนายภาพด้วย TFLite
def predict_image(interpreter, image):
    try:
        # ปรับขนาดภาพ
        img = image.resize((224, 224))
        # แปลงเป็น array และ preprocess สำหรับ VGG16
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)  # ใช้ preprocess_input ของ VGG16
        img_array = np.expand_dims(img_array, axis=0)  # เพิ่ม batch dimension

        # ตั้งค่า input tensor
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # ทำนาย
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        probabilities = prediction[0]

        return predicted_class, confidence, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Streamlit UI
st.title("Image Classification App with VGG16")
st.markdown("Upload an image to classify it using the VGG16 model (TensorFlow Lite).")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("Select an image...", type=['jpg', 'jpeg', 'png'])

# โหลดโมเดล TFLite
try:
    interpreter = tf.lite.Interpreter(model_path='vgg_model.tflite')
    interpreter.allocate_tensors()
    st.success("VGG16 TFLite model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load TFLite model: {e}")
    interpreter = None

# เมื่อผู้ใช้อัปโหลดภาพ
if uploaded_file is not None and interpreter is not None:
    try:
        # แสดงภาพที่อัปโหลด
        image = Image.open(uploaded_file).convert('RGB')  # แปลงเป็น RGB
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # ทำนาย
        with st.spinner('Predicting...'):
            predicted_class, confidence, probabilities = predict_image(interpreter, image)

        if predicted_class is not None:
            # แสดงผลการทำนาย
            st.success(f"**Predicted Class**: {predicted_class} (Confidence: {confidence:.2f})")

            # แสดงความน่าจะเป็นของทุกคลาสในรูปแบบตาราง
            st.write("**Probability of Each Class**:")
            prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            st.table(prob_dict)

            # พล็อตกราฟความน่าจะเป็น
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(class_names, probabilities)
            ax.set_xlabel('Class')
            ax.set_ylabel('Probability')
            ax.set_title('Probability of Each Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image to predict.")
