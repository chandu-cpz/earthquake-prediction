
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_filename = 'earthquake_model.pkl'
model = joblib.load(model_filename)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)

    # Convert the input data to a DataFrame
    df = pd.DataFrame([data])

    # Ensure the DataFrame has the same columns as the training data
    # You might need to add the same one-hot encoded columns here
    # For simplicity, let's assume the input data has the correct format

    # Make predictions
    prediction = model.predict(df)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
