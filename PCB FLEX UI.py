from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class UploadedImage(Base):
    __tablename__ = 'uploaded_images'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    image_path = Column(String)

# Create engine and session
db_path = os.path.join(os.path.dirname(__file__), 'pcb_database.db')
engine = create_engine(f'sqlite:///{db_path}', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Define a key for storing the history in the SessionState
HISTORY_KEY = "uploaded_images"
ADMIN_USERNAME = "TANSAM_TIDEL"
ADMIN_PASSWORD = "TANSAM123"

# Define the database connection and session
Base = declarative_base()
db_path = os.path.join(os.path.dirname(__file__), 'pcb_database.db')
engine = create_engine(f'sqlite:///{db_path}', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class UploadedImage(Base):
    __tablename__ = 'uploaded_images'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime)
    image_path = Column(String)

def main():
    st.title("PCB Board Class Predictor")
    st.markdown("Upload an image of a PCB board to predict its class.")

    # Load the session state
    session_state = st.session_state.get(HISTORY_KEY, [])
    admin_logged_in = st.session_state.get("admin_logged_in", False)

    if not admin_logged_in:
        if st.button("Admin Login", key="admin_login_button"):
            st.session_state["redirect"] = True
        if st.session_state.get("redirect", False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.session_state["redirect"] = False
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid username or password")
        else:
            file_uploaded = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])

            if file_uploaded is not None:
                try:
                    image = Image.open(file_uploaded)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    st.write("Classifying...")

                    result = predict_class(image)
                    if result == 'Non-Defective':
                        st.write(f"<h1 style='text-align: center; color: green;'>{result}</h1>", unsafe_allow_html=True)
                    else:
                        st.write(f"<h1 style='text-align: center; color: red;'><b>DEFECTIVE</b></h1>", unsafe_allow_html=True)
                        st.write(f"<h1 style='text-align: center; font-size: large;'>{result}</h1>", unsafe_allow_html=True)

                    # Add the uploaded image path and current date/time to the database
                    session = Session()
                    new_image = UploadedImage(datetime=datetime.now(), image_path=file_uploaded.name)
                    session.add(new_image)
                    session.commit()
                except Exception as e:
                    st.error("An error occurred during prediction.")
                    st.write(e)

            # Store the updated session state
            st.session_state[HISTORY_KEY] = session_state

# Display the history of uploaded images only if the admin is logged in
    if admin_logged_in:
        st.markdown("### Upload History", unsafe_allow_html=True)
        session = Session()
        history_data = [{"Date": item.datetime.strftime("%Y-%m-%d"), "Time": item.datetime.strftime("%H:%M:%S"), "Image": item.image_path, "Delete": st.checkbox(f"Delete {item.id}")} for item in session.query(UploadedImage).all()]
        
        delete_ids = [item["id"] for item in history_data if item["Delete"]]
        if st.button("Delete Selected") and delete_ids:
            session.query(UploadedImage).filter(UploadedImage.id.in_(delete_ids)).delete(synchronize_session=False)
            session.commit()
            st.success("Selected entries deleted successfully!")

        st.table(history_data)

        if st.button("Logout"):
            st.session_state["admin_logged_in"] = False

def predict_class(image):
    classifier_model = tf.keras.models.load_model(r"C:\Users\91934\Downloads\Pcb_cnn.h5")
    shape = (224, 224, 3)
    
    # Create a new model with hub.KerasLayer as the input layer
    input_layer = tf.keras.layers.Input(shape=shape)
    classifier_output = hub.KerasLayer(classifier_model)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=classifier_output)
    
    test_image = image.resize((224, 224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    
    class_names = ['Burnt', 'Cu pad Damaged', 'Non-Defective', 'Rust', 'Water Damaged']
    
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    
    return image_class

if __name__=="__main__":
    main()


