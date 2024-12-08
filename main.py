from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# File paths
DATA_PATH = 'C:/Users/Bhavana/PycharmProjects/PythonProject2/templates/cleaned_data.csv'
MODEL_PATH = 'C:/Users/Bhavana/PycharmProjects/PythonProject2/.venv/RidgeModel.pkl'

# Verify files
if not os.path.exists(DATA_PATH):
    print(f"Dataset not found at {DATA_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")

# Load the dataset and model
df = pd.read_csv(DATA_PATH)
pipe = pickle.load(open(MODEL_PATH, "rb"))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect the data from the form
            location = request.form.get('location')
            bhk = request.form.get('bhk')
            bath = request.form.get('bath')
            sqft = request.form.get('total_sqft')

            # Log the input values for debugging
            print(f"Raw Input: Location={location}, BHK={bhk}, Bath={bath}, SqFt={sqft}")

            # Validate inputs
            if not location or not bhk or not bath or not sqft:
                return render_template('index.html', prediction="All fields are required.",
                                       locations=sorted(df['location'].unique()))

            try:
                bhk = float(bhk)
                bath = float(bath)
                sqft = float(sqft)
            except ValueError:
                return render_template('index.html', prediction="BHK, Bathrooms, and Square Feet must be numbers.",
                                       locations=sorted(df['location'].unique()))

            # Prepare input data as DataFrame
            input_data = pd.DataFrame([[location, bhk, bath, sqft]], columns=['location', 'bhk', 'bath', 'sqft'])

            # One-hot encode the location feature using the same approach as during model training
            encoded_location = pd.get_dummies(input_data['location'], prefix='location')

            # Concatenate the encoded location with the rest of the input data
            input_data = pd.concat([input_data.drop('location', axis=1), encoded_location], axis=1)

            # Log the DataFrame after one-hot encoding for debugging
            print("Encoded Input DataFrame for Prediction:")
            print(input_data)

            # Ensure all the columns required by the model are present
            required_columns = pipe.feature_names_in_
            print(f"Model expects these features: {required_columns}")

            # Add missing columns with default value 0
            for col in required_columns:
                if col not in input_data.columns:
                    print(f"Adding missing column: {col}")
                    input_data[col] = 0

            # Re-order the input data to match the feature order expected by the model
            input_data = input_data[required_columns]

            # Log transformed input for debugging
            print("Transformed Input DataFrame for Prediction:")
            print(input_data)

            # Make prediction using the model
            prediction = pipe.predict(input_data)[0] * 100000  # Multiply by 100000 for scaling

            # Log the prediction
            print(f"Predicted Price: ₹{prediction}")

            # Return the predicted price
            return render_template('index.html', prediction=np.round(prediction, 2),
                                   locations=sorted(df['location'].unique()))

        except Exception as e:
            # Detailed error logging
            print(f"Error during prediction: {e}")
            return render_template('index.html', prediction=f"Error in prediction: {e}",
                                   locations=sorted(df['location'].unique()))

    else:
        # For GET request, just load the form
        locations = sorted(df['location'].unique())
        return render_template('index.html', locations=locations)


if __name__ == '__main__':
    app.run(debug=True)






