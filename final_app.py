import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"F:\New folder (2)\trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Remedies dictionary
remedies = {
     "Apple___Apple_scab": "Use fungicides like captan or myclobutanil. Prune affected areas.",
     "Apple___Black_rot": "Remove and destroy infected branches. Apply copper fungicides.",
     "Apple___Cedar_apple_rust": "Use sulfur sprays. Plant resistant varieties.",
     "Apple___healthy": "No treatment required. Keep monitoring.",
     "Blueberry___healthy": "No treatment required. Ensure proper care.",
     "Cherry_(including_sour)___Powdery_mildew": "Apply potassium bicarbonate sprays. Ensure proper ventilation.",
     "Cherry_(including_sour)___healthy": "No treatment required. Maintain good soil health.",
     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops and apply fungicides like azoxystrobin.",
     "Corn_(maize)___Common_rust_": "Use resistant hybrids. Apply fungicides if necessary.",
     "Corn_(maize)___Northern_Leaf_Blight": "Apply strobilurin fungicides. Ensure crop rotation.",
     "Corn_(maize)___healthy": "No treatment required. Keep monitoring.",
     "Grape___Black_rot": "Prune infected areas. Use fungicides like mancozeb.",
     "Grape___Esca_(Black_Measles)": "Prune infected vines. Apply fungicides as necessary.",
     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides. Maintain good air circulation.",
     "Grape___healthy": "No treatment required. Ensure proper care.",
     "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees. Use insecticides to control psyllids.",
     "Peach___Bacterial_spot": "Use copper-based sprays. Avoid overhead irrigation.",
     "Peach___healthy": "No treatment required. Keep monitoring.",
     "Pepper,_bell___Bacterial_spot": "Use copper-based sprays. Ensure good ventilation.",
     "Pepper,_bell___healthy": "No treatment required. Maintain good care practices.",
     "Potato___Early_blight": "Apply chlorothalonil-based fungicides. Remove infected leaves.",
     "Potato___Late_blight": "Use metalaxyl fungicides. Avoid overwatering.",
     "Potato___healthy": "No treatment required. Maintain regular care.",
     "Raspberry___healthy": "No treatment required. Keep monitoring.",
     "Soybean___healthy": "No treatment required. Maintain good soil health.",
     "Squash___Powdery_mildew": "Apply sulfur or potassium bicarbonate sprays.",
     "Strawberry___Leaf_scorch": "Remove infected leaves. Use fungicides like captan.",
     "Strawberry___healthy": "No treatment required. Keep monitoring.",
     "Tomato___Bacterial_spot": "Use copper-based sprays. Avoid handling wet plants.",
     "Tomato___Early_blight": "Apply fungicides like chlorothalonil. Rotate crops.",
     "Tomato___Late_blight": "Use fungicides containing metalaxyl. Avoid overcrowding plants.",
     "Tomato___Leaf_Mold": "Ensure good ventilation. Apply fungicides like mancozeb.",
     "Tomato___Septoria_leaf_spot": "Prune infected leaves. Use fungicides like chlorothalonil.",
     "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides or insecticidal soaps.",
     "Tomato___Target_Spot": "Apply fungicides like azoxystrobin. Remove infected leaves.",
     "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies. Remove infected plants.",
     "Tomato___Tomato_mosaic_virus": "Remove infected plants. Sterilize tools regularly.",
     "Tomato___healthy": "No treatment required. Keep monitoring."
}


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Remedies", "Visualizations"])


# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNIZER")
    image_path = r"F:\New folder (2)\images\home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognizer System! 🌿🔍
     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on Kagle.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                 ### Why Choose Us?
                - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
                - **User-Friendly:** Simple and intuitive interface for seamless user experience.
                - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
                """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    if st.button("Recognize"):
        st.snow()
        st.write("Our Recognization")
        result_index = model_prediction(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        prediction = class_name[result_index]
        st.success(f"Model is Predicting it's a {prediction}")
        st.markdown(f"**Remedy:** {remedies.get(prediction, 'No remedy available')}")

elif app_mode == "Remedies":
    st.header("Remedies for Plant Diseases")
    for disease, remedy in remedies.items():
        st.subheader(disease)
        st.write(remedy)

elif app_mode == "Visualizations":
    st.title("Disease Trends and Insights")

    # Bar Chart for Disease Distribution
    disease_data = pd.DataFrame({
        "Disease": ["Apple Scab", "Black Rot", "Powdery Mildew", "Late Blight"],
        "Occurrences": [30, 45, 50, 25]
    })
    st.subheader("Disease Distribution")
    fig = px.bar(disease_data, x="Disease", y="Occurrences", title="Occurrences of Diseases")
    st.plotly_chart(fig)

    # Line Chart for Training vs Validation
    st.subheader("Model Accuracy and Loss")
    line_data = pd.DataFrame({
        "Epoch": [1, 2, 3, 4, 5],
        "Training Accuracy": [0.7, 0.8, 0.85, 0.9, 0.92],
        "Validation Accuracy": [0.68, 0.78, 0.82, 0.88, 0.9],
        "Training Loss": [0.6, 0.5, 0.4, 0.3, 0.2],
        "Validation Loss": [0.65, 0.55, 0.45, 0.35, 0.25]
    })

    accuracy_fig = px.line(line_data, x="Epoch", y=["Training Accuracy", "Validation Accuracy"],
                           title="Accuracy Over Epochs")
    st.plotly_chart(accuracy_fig)

    loss_fig = px.line(line_data, x="Epoch", y=["Training Loss", "Validation Loss"],
                       title="Loss Over Epochs")
    st.plotly_chart(loss_fig)
