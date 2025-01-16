# Lung Cancer Detection 

This project utilizes Deep Learning techniques to classify lung cancer from medical images (e.g., X-rays, CT scans). The model is trained using convolutional neural networks (CNNs) to detect cancerous cells in lung images. A Flask-based web interface is provided, where users can upload images for classification. The results are displayed on the web page after the model processes the image.

## Installation

To run the project locally, follow these steps:

### Prerequisites

- Python 3.x
- pip (Python package manager)
- Basic knowledge of Flask and HTML

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/lung-cancer-detection.git
    cd lung-cancer-detection
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    venv\Scripts\activate  # For Windows
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Train the Deep Learning Model (if not already trained):**

    - You can train the model using your dataset of lung X-ray or CT scan images. The model is typically built using a Convolutional Neural Network (CNN).
    - If you have a pre-trained model, skip this step and load it directly for prediction.

5. **Run the Flask app:**

    Start the Flask server by running the following command:

    ```bash
    python app.py
    ```

    The Flask app will run on `http://localhost:5000`.

## Usage

1. Open your browser and navigate to the provided local URL (typically `http://localhost:5000`).
2. The web interface will allow users to upload images for classification.
3. Once the image is uploaded, the model will classify the image and display the result (e.g., "Cancer Detected" or "No Cancer Detected").

## Screenshot of the Web Interface

Here’s a preview of the web interface for the Lung Cancer Detection project:

![Lung Cancer Detection Interface](https://github.com/user-attachments/assets/d9dd6340-052d-44f7-aacf-b1fe02559185)

## Flask Web Interface

The web interface is developed using Flask and HTML/CSS. It allows users to upload images and display the result.

### HTML Template (upload.html):

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Detection</title>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="container">
        <h1>Lung Cancer Detection</h1>
        <form action="/" method="POST" enctype="multipart/form-data">
            <label for="image">Upload an Image</label>
            <input type="file" name="image" required>
            <button type="submit">Upload</button>
        </form>
        {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
        {% endif %}
    </div>
</body>
</html>
