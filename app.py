from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = [float(x) for x in request.form.values()]
        print(f"Received input data: {data}")  # Debugging line

        # Ensure correct number of features
        if len(data) != 9:
            return jsonify({"error": f"Expected 9 features, but received {len(data)}"})

        # Convert to NumPy array and reshape for model input
        features = np.array([data])

        # Make prediction
        prediction = model.predict(features)
        result = "Malignant" if prediction[0] == 1 else "Benign"

        return render_template("index.html", prediction_text=f"Diagnosis: {result}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

