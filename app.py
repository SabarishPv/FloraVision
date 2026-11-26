
import os
import json
import time
import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr

from pathlib import Path


# ==================================================
# 1. Project Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = MODEL_DIR / "final_flower_classifier.keras"


# ==================================================
# 2. Load Trained Model and Metadata
# ==================================================

print("Loading FloraVision model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(MODEL_DIR / "class_names.json") as f:
    class_names = json.load(f)

with open(MODEL_DIR / "model_metadata.json") as f:
    metadata = json.load(f)

IMG_HEIGHT, IMG_WIDTH = metadata["input_shape"][:2]

assert model.output_shape[-1] == len(class_names)

print("Model loaded successfully!")
print("Architecture:", metadata["model_name"])
print("Flower categories:", len(class_names))


# ==================================================
# 3. Image Preprocessing and Prediction
# ==================================================

def classify_flower(image):

    if image is None:
        raise gr.Error("Please upload a flower image.")

    start = time.perf_counter()

    image = image.convert("RGB")

    image_array = tf.keras.utils.img_to_array(image)

    image_array = tf.image.resize(
        image_array,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    image_array = image_array / 255.0

    input_tensor = tf.expand_dims(image_array, axis=0)

    probabilities = model(
        input_tensor,
        training=False
    ).numpy()[0]

    top_indices = np.argsort(probabilities)[-5:][::-1]

    predictions = {
        class_names[i].replace("_", " ").title():
        float(probabilities[i])
        for i in top_indices
    }

    best_index = top_indices[0]

    flower = class_names[best_index].replace(
        "_", " "
    ).title()

    confidence = float(probabilities[best_index] * 100)

    elapsed = (time.perf_counter() - start) * 1000

    status = (
        "High" if confidence >= 80
        else "Moderate" if confidence >= 50
        else "Low"
    )

    summary = f"""
### Classification Result

**Predicted Flower:** {flower}

**Confidence:** {confidence:.2f}%

**Confidence Level:** {status}

**Processing Time:** {elapsed:.2f} ms

**Architecture:** {metadata["model_name"]}
"""

    table = pd.DataFrame({
        "Rank": range(1, len(top_indices) + 1),
        "Flower": [
            class_names[i].replace("_", " ").title()
            for i in top_indices
        ],
        "Confidence (%)": [
            round(float(probabilities[i] * 100), 2)
            for i in top_indices
        ]
    })

    return predictions, summary, table


# ==================================================
# 4. User Interface Styling
# ==================================================

css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

#header {
    background: linear-gradient(
        135deg, #14532d, #166534, #15803d
    );
    padding: 32px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
}

#header h1 {
    color: white !important;
    font-size: 38px;
    margin: 0;
}

#header p {
    color: #dcfce7 !important;
    margin-top: 10px;
}

#predict-btn {
    background: #166534 !important;
    color: white !important;
    min-height: 48px;
    border-radius: 10px;
}

#footer {
    text-align: center;
    padding: 20px;
    opacity: 0.7;
}
"""

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate"
)


# ==================================================
# 5. Gradio Application
# ==================================================

blocks_options = {
    "title": "FloraVision | Flower Classification"
}

launch_options = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
    "inbrowser": True
}

# Support different Gradio versions
if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_options.update(theme=theme, css=css)
else:
    launch_options.update(theme=theme, css=css)


with gr.Blocks(**blocks_options) as demo:

    gr.HTML("""
    <div id="header">
        <h1>FloraVision</h1>
        <p>Intelligent Flower Classification System</p>
        <p>Deep Learning | Computer Vision | Transfer Learning</p>
    </div>
    """)

    # ----------------------------------------------
    # Tab 1: Flower Classification
    # ----------------------------------------------

    with gr.Tab("Flower Classification"):

        gr.Markdown(
            "### Upload a flower photograph to identify its category."
        )

        with gr.Row():

            with gr.Column():

                image_input = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="Flower Image",
                    height=350
                )

                with gr.Row():

                    predict_btn = gr.Button(
                        "Classify Flower",
                        variant="primary",
                        elem_id="predict-btn"
                    )

                    clear_btn = gr.Button("Clear")

            with gr.Column():

                prediction_output = gr.Label(
                    num_top_classes=5,
                    label="Top-5 Predictions"
                )

                result_output = gr.Markdown(
                    "Upload an image to begin."
                )

        table_output = gr.Dataframe(
            headers=[
                "Rank",
                "Flower",
                "Confidence (%)"
            ],
            interactive=False,
            label="Detailed Classification Results"
        )

        gr.Markdown("""
        **Note:** Predictions are limited to supported flower
        categories. Confidence does not guarantee correctness.
        """)

    # ----------------------------------------------
    # Tab 2: Model Performance
    # ----------------------------------------------

    with gr.Tab("Model Performance"):

        gr.Markdown(f"""
        ## Final Model Evaluation

        | Metric | Result |
        |---|---|
        | Architecture | {metadata["model_name"]} |
        | Flower Categories | {len(class_names)} |
        | Parameters | {metadata["parameters"]:,} |
        | Validation Accuracy | {metadata["validation_accuracy"]:.2f}% |
        | Test Accuracy | {metadata["test_accuracy"]:.2f}% |
        | Macro F1-Score | {metadata["macro_f1_score"]:.2f}% |
        | Top-5 Accuracy | {metadata["top5_accuracy"]:.2f}% |
        | Image Resolution | {IMG_HEIGHT} x {IMG_WIDTH} |
        """)

        comparison_path = (
            RESULTS_DIR / "final_architecture_comparison.csv"
        )

        if comparison_path.exists():

            comparison = pd.read_csv(comparison_path)

            comparison = comparison.drop(
                columns=["File"],
                errors="ignore"
            )

            gr.Dataframe(
                value=comparison.round(4),
                interactive=False,
                label="Architecture Comparison"
            )

        confusion_path = RESULTS_DIR / "confusion_matrix.png"

        if confusion_path.exists():

            gr.Image(
                value=str(confusion_path),
                label="Confusion Matrix",
                interactive=False
            )

    # ----------------------------------------------
    # Tab 3: Supported Flower Categories
    # ----------------------------------------------

    with gr.Tab("Flower Categories"):

        gr.Markdown(
            f"## {len(class_names)} Supported Flower Categories"
        )

        categories = pd.DataFrame({
            "Class ID": range(len(class_names)),
            "Flower Name": [
                name.replace("_", " ").title()
                for name in class_names
            ]
        })

        gr.Dataframe(
            value=categories,
            interactive=False
        )

    # ----------------------------------------------
    # Tab 4: About Project
    # ----------------------------------------------

    with gr.Tab("About Project"):

        gr.Markdown("""
        # About FloraVision

        FloraVision is an end-to-end deep learning
        system for multi-class flower classification.

        ## Datasets

        - TensorFlow Flower Photos
        - Oxford 102 Flowers

        ## Neural Network Architectures

        - Baseline Linear Neural Network
        - Hyperparameter-Optimized LNN
        - Deep Neural Network
        - Convolutional Neural Network
        - MobileNetV2 Transfer Learning
        - Fine-Tuned MobileNetV2

        ## Project Workflow

        1. Dataset collection and integration
        2. Data cleaning and duplicate detection
        3. RGB conversion and image resizing
        4. Pixel normalization
        5. Stratified dataset splitting
        6. Batch dataset creation
        7. Neural network training
        8. Hyperparameter tuning
        9. Data augmentation
        10. Transfer learning and fine-tuning
        11. Model evaluation and comparison
        12. Gradio application deployment

        ## Technology Stack

        Python, TensorFlow, Keras, NumPy, Pandas,
        Matplotlib, Scikit-learn, Pillow and Gradio.

        ## Limitations

        - Similar flower species may be confused.
        - Unknown flower categories are not detected.
        - Lighting and image quality affect predictions.
        - Confidence scores are not calibrated.
        """)

    # ----------------------------------------------
    # Event Handlers
    # ----------------------------------------------

    predict_btn.click(
        fn=classify_flower,
        inputs=image_input,
        outputs=[
            prediction_output,
            result_output,
            table_output
        ]
    )

    clear_btn.click(
        fn=lambda: (
            None,
            None,
            "Upload an image to begin.",
            None
        ),
        inputs=None,
        outputs=[
            image_input,
            prediction_output,
            result_output,
            table_output
        ]
    )

    gr.HTML("""
    <div id="footer">
        FloraVision — Multi-Class Flower Classification
        <br>
        Powered by TensorFlow, Keras and Gradio
    </div>
    """)


# ==================================================
# 6. Launch Local Application
# ==================================================

if __name__ == "__main__":
    demo.launch(**launch_options)
