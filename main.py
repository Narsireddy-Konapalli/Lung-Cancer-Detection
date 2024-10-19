from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

# Load the pre-trained model
h5_file_path = 'lung_cancer_model.h5'
model = load_model(h5_file_path)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maximum file size: 16MB

# Categories for predictions
cat = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Form Class
class ImageUploadForm(FlaskForm):
    image = FileField('Upload an Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    submit = SubmitField('Submit')

# Helper function to convert image to base64
def pil_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

# Route for image upload form
@app.route('/home', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = ImageUploadForm()
    prediction = None
    image_data = None

    if form.validate_on_submit():
        # Process the uploaded image
        image = form.image.data
        if image:
            pil_image = Image.open(image)
            image_cv = np.array(pil_image)
            if image_cv.shape[-1] != 3:
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)

            # Resize and normalize the image
            resized_image = cv2.resize(image_cv, (400, 400))  
            nor_img = resized_image / 255.0 
            array_4d = np.expand_dims(nor_img, axis=0) 

            # Set your confidence threshold
            confidence_threshold = 0.7

            test_predicted = model.predict(array_4d)
            max_probability = np.max(test_predicted)
            prediction_label = cat[np.argmax(test_predicted)]

            # Check if the maximum probability exceeds the confidence threshold
            if max_probability >= confidence_threshold:
                prediction = prediction_label
            else:
                prediction = "Cannot be determined"

            # Convert the image to base64 to embed it directly in HTML
            image_data = pil_image_to_base64(pil_image)

    return render_template('index.html', form=form, prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
