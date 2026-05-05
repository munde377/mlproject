from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# ---------------- HOME PAGE ----------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- PREDICT PAGE ----------------
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    # 👉 When user directly opens /predictdata
    if request.method == "GET":
        return render_template("home.html", results=None)

    try:
        # 👉 Get form data safely
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        pred_df = data.get_data_as_data_frame()
        print("Input Data:\n", pred_df)

        # 👉 Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        final_result = round(results[0], 2)

        print("Prediction:", final_result)

        return render_template("home.html", results=final_result)

    except Exception as e:
        print("Error:", e)
        return render_template("home.html", results="Error occurred")


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = 5000
    print(f"👉 Open this URL in browser: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)