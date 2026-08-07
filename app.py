from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input

from preprocessing import crop_retina, apply_clahe

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

def get_model():
    global model

    if model is None:
        model = tf.keras.models.load_model(
            "model/EfficientNetB0_Best.keras"
        )

    return model

classes = {

    0: "No DR",

    1: "Mild",

    2: "Moderate",

    3: "Severe",

    4: "Proliferative DR"

}

descriptions = {

    "No DR":
    "No diabetic retinopathy detected.",

    "Mild":
    "Mild diabetic retinopathy detected. Regular eye examination is recommended.",

    "Moderate":
    "Moderate diabetic retinopathy detected. Please consult an ophthalmologist.",

    "Severe":
    "Severe diabetic retinopathy detected. Immediate specialist consultation is advised.",

    "Proliferative DR":
    "Proliferative diabetic retinopathy detected. Urgent medical attention is required."

}


def prepare_image(path):

    image = cv2.imread(path)

    image = crop_retina(image)

    image = apply_clahe(image)

    image = cv2.resize(image, (224,224))

    processed = image.copy()

    image = image.astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image, processed


@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/test")
def test():
    return render_template("test.html", active="test")


@app.route("/team")
def team():
    return render_template("team.html", active="team")


@app.route("/contact")
def contact():
    return render_template("contact.html", active="contact")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return "No Image Uploaded"

    file = request.files["image"]

    filename = file.filename

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )

    file.save(filepath)

    image, processed = prepare_image(filepath)

    prediction = get_model().predict(image)

    pred_class = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    processed_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        "processed_" + filename

    )

    cv2.imwrite(processed_path, processed)

    return render_template(

        "result.html",

        active="test",

        original=filepath,

        processed=processed_path,

        prediction=classes[pred_class],

        confidence=round(confidence,2),

        description=descriptions[classes[pred_class]]

    )


if __name__ == "__main__":

    app.run(debug=True)
