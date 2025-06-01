from flask import Flask, jsonify, render_template, request , url_for
import pickle
import numpy as np
import logging
import requests # Import requests for making HTTP calls to Gemini API
import json # Import json for handling JSON data

# Initialize Flask app
app = Flask(__name__)

# Configure basic logging for better error visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Models (No Scalers) ---
# IMPORTANT: This version of app.py assumes your machine learning models
# (diabetes_model.pkl, heart_model.pkl, lung_model.pkl) were trained on
# UNSEALED/RAW data. If your models were trained using a StandardScaler, MinMaxScaler,
# or any other scaling technique, you MUST revert to the previous app.py version
# that loads and applies the corresponding scaler. Running models trained on scaled
# data with unscaled input will lead to inaccurate predictions.
try:
    # Load Diabetes Model
    diabetes_model = pickle.load(open('model_files/diabetes_model.pkl', 'rb'))
    app.logger.info("Diabetes model loaded successfully.")

    # Load Heart Disease Model
    heart_model = pickle.load(open('model_files/heart_model.pkl', 'rb'))
    app.logger.info("Heart disease model loaded successfully.")

    # Load Lung Disease Model
    lung_model = pickle.load(open('model_files/lung_model.pkl', 'rb'))
    app.logger.info("Lung disease model loaded successfully.")

except FileNotFoundError as e:
    app.logger.error(f"Error: Model file not found. Please ensure 'model_files' directory exists and contains all .pkl files. Details: {e}")
    # If critical files are missing, the application cannot function. Exit gracefully.
    exit("Failed to load ML models. Please check 'model_files' directory and file names.")
except Exception as e:
    app.logger.error(f"An unexpected error occurred during model loading: {e}")
    exit("Failed to load ML models due to an unexpected error.")

# --- Health Tips Function ---
def get_health_tips(disease_predictions):
    """
    Generates health tips based on the predicted disease risks.

    Args:
        disease_predictions (dict): A dictionary where keys are disease names (e.g., 'diabetes', 'heart', 'lung')
                                    and values are their prediction status (0 for low risk, 1 for high risk).

    Returns:
        list: A list of relevant health tips.
    """
    tips = []

    if disease_predictions.get('diabetes') == 1:
        tips.append("Diabetes: It's highly recommended to consult a healthcare professional for a confirmed diagnosis and personalized advice. Focus on a balanced diet, regular exercise, and consistent blood sugar monitoring.")
    elif disease_predictions.get('diabetes') == 0:
        tips.append("Diabetes: Your risk is currently low based on the provided data. Continue to maintain a healthy lifestyle with a balanced diet and regular exercise to further reduce future risk.")

    if disease_predictions.get('heart') == 1:
        tips.append("Heart Disease: Seek immediate medical advice for potential heart disease. Adopt a heart-healthy diet, manage stress effectively, and ensure regular cardiovascular check-ups.")
    elif disease_predictions.get('heart') == 0:
        tips.append("Heart Disease: Your risk is currently low. Keep up a healthy lifestyle including regular physical activity, a balanced diet, and avoiding smoking.")

    if disease_predictions.get('lung') == 1: # Assuming 'lung' is the key for lung cancer
        tips.append("Lung Disease: Consult a doctor immediately for evaluation. If you smoke, quitting is the single most important step. Avoid exposure to air pollutants and undergo regular health screenings.")
    elif disease_predictions.get('lung') == 0:
        tips.append("Lung Disease: Your risk is currently low. Continue to avoid smoking, secondhand smoke, and ensure good indoor air quality to protect your lungs.")

    if not tips:
        tips.append("No specific risks detected based on your input. Remember to maintain a healthy lifestyle, stay hydrated, and have regular medical check-ups for overall well-being.")

    return tips

# --- Flask Routes ---

@app.route('/')
def home():
    """
    Renders the main index page of the application.
    This page will contain iframes to load individual disease prediction forms.
    """
    return render_template('index.html')

@app.route('/get_prediction_content/<disease_type>')
def get_prediction_content(disease_type):
    """
    Dynamically loads and serves HTML content for different disease prediction forms.
    This route is now specifically used by the `<iframe>` elements in `index.html`
    to load the content of each disease's form page.

    Args:
        disease_type (str): The type of disease (e.g., 'diabetes', 'heart_disease', 'lung_disease').

    Returns:
        Response: Rendered HTML template or an error message.
    """
    if disease_type == 'diabetes':
        return render_template('diabetes.html')
    elif disease_type == 'heart_disease':
        return render_template('heart.html')
    elif disease_type == 'lung_disease':
        return render_template('lung_cancer.html')
    else:
        app.logger.warning(f"Requested content for unknown disease type: {disease_type}")
        return "<h3>Content not found</h3>", 404

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """
    Handles POST requests for diabetes prediction.
    Expects JSON data from the frontend form (diabetes.html).
    """
    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)
        app.logger.info(f"Received diabetes prediction request with data: {data}")

        # Define the exact feature keys and their expected order for the diabetes model
        # These keys must match the 'name' attributes of your input fields in diabetes.html
        diabetes_feature_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        diabetes_features_raw = []

        # Validate and collect features
        for key in diabetes_feature_keys:
            if key not in data or data[key] is None:
                app.logger.error(f"Missing data for diabetes feature: {key}")
                return jsonify({'error': f'Missing data for diabetes feature: "{key}". Please provide all required information.'}), 400
            try:
                diabetes_features_raw.append(float(data[key]))
            except ValueError:
                app.logger.error(f"Invalid data type for diabetes feature '{key}'. Expected a number.")
                return jsonify({'error': f'Invalid data type for diabetes feature "{key}". Please enter a valid number.'}), 400

        # Convert the list of features to a NumPy array and reshape for single prediction
        # .reshape(1, -1) converts a 1D array into a 2D array with 1 row, suitable for scikit-learn models
        final_features = np.array(diabetes_features_raw).reshape(1, -1)

        # No scaler.transform() call here, assuming model trained on unscaled data
        prediction_proba = diabetes_model.predict_proba(final_features)[0]
        risk_percentage = round(prediction_proba[1] * 100, 2) # Round to 2 decimal places

        prediction_status = 1 if risk_percentage >= 50 else 0

        tips = get_health_tips({'diabetes': prediction_status})

        app.logger.info(f"Diabetes prediction result: {prediction_status}, Risk: {risk_percentage}%, Tips: {tips}")
        return jsonify(prediction=prediction_status, risk_percentage=risk_percentage, tips=tips)

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during diabetes prediction: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while processing your diabetes prediction. Please try again later.'}), 500

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    """
    Handles POST requests for heart disease prediction.
    Expects JSON data from the frontend form (heart.html).
    """
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received heart disease prediction request with data: {data}")

        heart_feature_keys = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        heart_features_raw = []

        for key in heart_feature_keys:
            if key not in data or data[key] is None:
                app.logger.error(f"Missing data for heart feature: {key}")
                return jsonify({'error': f'Missing data for heart feature: "{key}". Please provide all required information.'}), 400
            try:
                heart_features_raw.append(float(data[key]))
            except ValueError:
                app.logger.error(f"Invalid data type for heart feature '{key}'. Expected a number.")
                return jsonify({'error': f'Invalid data type for heart feature "{key}". Please enter a valid number.'}), 400

        final_features = np.array(heart_features_raw).reshape(1, -1)

        prediction_proba = heart_model.predict_proba(final_features)[0]
        risk_percentage = round(prediction_proba[1] * 100, 2)

        prediction_status = 1 if risk_percentage >= 50 else 0

        tips = get_health_tips({'heart': prediction_status})

        app.logger.info(f"Heart disease prediction result: {prediction_status}, Risk: {risk_percentage}%, Tips: {tips}")
        return jsonify(prediction=prediction_status, risk_percentage=risk_percentage, tips=tips)

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during heart disease prediction: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while processing your heart disease prediction. Please try again later.'}), 500

@app.route('/predict/lung', methods=['POST'])
def predict_lung():
    """
    Handles POST requests for lung disease prediction.
    Expects JSON data from the frontend form (lung_cancer.html).
    """
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received lung disease prediction request with data: {data}")

        lung_feature_keys = [
            'Gender', 'Age', 'Smoking', 'Yellow_Fingers', 'Anxiety', 'Peer_Pressure',
            'Chronic_Disease', 'Fatigue', 'Allergy', 'Wheezing', 'Alcohol_Consuming',
            'Coughing', 'Shortness_of_Breath', 'Swallowing_Difficulty', 'Chest_Pain'
        ]
        lung_features_raw = []

        for key in lung_feature_keys:
            if key not in data or data[key] is None:
                app.logger.error(f"Missing data for lung feature: {key}")
                return jsonify({'error': f'Missing data for lung feature: "{key}". Please provide all required information.'}), 400
            try:
                lung_features_raw.append(float(data[key]))
            except ValueError:
                app.logger.error(f"Invalid data type for lung feature '{key}'. Expected a number.")
                return jsonify({'error': f'Invalid data type for lung feature "{key}". Please enter a valid number.'}), 400

        final_features = np.array(lung_features_raw).reshape(1, -1)

        prediction_proba = lung_model.predict_proba(final_features)[0]
        risk_percentage = round(prediction_proba[1] * 100, 2)

        prediction_status = 1 if risk_percentage >= 50 else 0

        tips = get_health_tips({'lung': prediction_status})

        app.logger.info(f"Lung disease prediction result: {prediction_status}, Risk: {risk_percentage}%, Tips: {tips}")
        return jsonify(prediction=prediction_status, risk_percentage=risk_percentage, tips=tips)

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during lung disease prediction: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while processing your lung disease prediction. Please try again later.'}), 500

# --- LLM Integration: New Route for Gemini API Calls ---
@app.route('/get_llm_insight', methods=['POST'])
def get_llm_insight():
    """
    Handles POST requests to generate LLM insights based on disease prediction data.
    """
    try:
        request_data = request.get_json(force=True)
        disease_type = request_data.get('disease_type')
        prediction_data = request_data.get('prediction_data', {})

        app.logger.info(f"Received LLM insight request for {disease_type} with data: {prediction_data}")

        # Map numerical/coded values to human-readable strings for the LLM prompt
        # This makes the prompt more natural for the LLM to understand.
        human_readable_data = {}
        if disease_type == 'diabetes':
            # Diabetes features are mostly numerical, no complex mapping needed beyond direct values
            human_readable_data = {k: v for k, v in prediction_data.items() if k != 'prediction' and k != 'risk_percentage'}
        elif disease_type == 'heart':
            # Map heart disease categorical values
            sex_map = {1: 'Male', 0: 'Female'}
            cp_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}
            fbs_map = {1: 'Greater than 120 mg/ml', 0: 'Lower than 120 mg/ml'}
            restecg_map = {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}
            exang_map = {1: 'Yes', 0: 'No'}
            slope_map = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
            ca_map = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three'}
            thal_map = {0: 'Unknown', 1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}

            human_readable_data['Age'] = prediction_data.get('age')
            human_readable_data['Sex'] = sex_map.get(int(prediction_data.get('sex')), 'N/A')
            human_readable_data['Chest Pain Type'] = cp_map.get(int(prediction_data.get('cp')), 'N/A')
            human_readable_data['Resting Blood Pressure'] = prediction_data.get('trestbps')
            human_readable_data['Cholesterol'] = prediction_data.get('chol')
            human_readable_data['Fasting Blood Sugar > 120 mg/dl'] = fbs_map.get(int(prediction_data.get('fbs')), 'N/A')
            human_readable_data['Resting ECG Results'] = restecg_map.get(int(prediction_data.get('restecg')), 'N/A')
            human_readable_data['Max Heart Rate Achieved'] = prediction_data.get('thalach')
            human_readable_data['Exercise Induced Angina'] = exang_map.get(int(prediction_data.get('exang')), 'N/A')
            human_readable_data['Oldpeak'] = prediction_data.get('oldpeak')
            human_readable_data['Slope of Peak Exercise ST Segment'] = slope_map.get(int(prediction_data.get('slope')), 'N/A')
            human_readable_data['Vessels Colored by Flourosopy'] = ca_map.get(int(prediction_data.get('ca')), 'N/A')
            human_readable_data['Thalassemia'] = thal_map.get(int(prediction_data.get('thal')), 'N/A')

        elif disease_type == 'lung':
            # Map lung disease categorical values (assuming 0/1 for Yes/No)
            binary_map = {1: 'Yes', 0: 'No'}
            human_readable_data['Gender'] = binary_map.get(int(prediction_data.get('Gender')), 'N/A') # Assuming Gender is 0/1
            human_readable_data['Age'] = prediction_data.get('Age')
            human_readable_data['Smoking'] = binary_map.get(int(prediction_data.get('Smoking')), 'N/A')
            human_readable_data['Yellow Fingers'] = binary_map.get(int(prediction_data.get('Yellow_Fingers')), 'N/A')
            human_readable_data['Anxiety'] = binary_map.get(int(prediction_data.get('Anxiety')), 'N/A')
            human_readable_data['Peer Pressure'] = binary_map.get(int(prediction_data.get('Peer_Pressure')), 'N/A')
            human_readable_data['Chronic Disease'] = binary_map.get(int(prediction_data.get('Chronic_Disease')), 'N/A')
            human_readable_data['Fatigue'] = binary_map.get(int(prediction_data.get('Fatigue')), 'N/A')
            human_readable_data['Allergy'] = binary_map.get(int(prediction_data.get('Allergy')), 'N/A')
            human_readable_data['Wheezing'] = binary_map.get(int(prediction_data.get('Wheezing')), 'N/A')
            human_readable_data['Alcohol Consuming'] = binary_map.get(int(prediction_data.get('Alcohol_Consuming')), 'N/A')
            human_readable_data['Coughing'] = binary_map.get(int(prediction_data.get('Coughing')), 'N/A')
            human_readable_data['Shortness of Breath'] = binary_map.get(int(prediction_data.get('Shortness_of_Breath')), 'N/A')
            human_readable_data['Swallowing Difficulty'] = binary_map.get(int(prediction_data.get('Swallowing_Difficulty')), 'N/A')
            human_readable_data['Chest Pain'] = binary_map.get(int(prediction_data.get('Chest_Pain')), 'N/A')


        # Construct the prompt for the LLM
        risk_status = "High Risk" if prediction_data.get('prediction') == 1 else "Low Risk"
        risk_percentage = prediction_data.get('risk_percentage')

        prompt_parts = [
            f"A user has received a {risk_status} prediction for {disease_type.replace('_', ' ')} with a risk percentage of {risk_percentage}%."
        ]

        if human_readable_data:
            prompt_parts.append("Here are the user's input factors:")
            for key, value in human_readable_data.items():
                prompt_parts.append(f"- {key}: {value}")

        if risk_status == "High Risk":
            prompt_parts.append("Based on this, provide general, non-medical advice on lifestyle adjustments, common preventative measures, and next steps. Emphasize the importance of consulting a qualified healthcare professional for a confirmed diagnosis and personalized advice. Do NOT give medical advice or diagnose.")
        else:
            prompt_parts.append("Based on this, provide general tips for maintaining good health and preventing the disease. Emphasize the importance of consulting a qualified healthcare professional for any health concerns. Do NOT give medical advice or diagnose.")

        prompt_parts.append("Keep the response concise, informative, and encouraging. Start with a friendly greeting.")
        
        llm_prompt = "\n".join(prompt_parts)
        app.logger.info(f"LLM Prompt: {llm_prompt}")

        # Call the Gemini API
        api_key = "############" # Leave this empty. Canvas will inject the API key at runtime.
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": llm_prompt}]}
            ]
        }

        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        llm_result = response.json()
        
        if llm_result and 'candidates' in llm_result and len(llm_result['candidates']) > 0:
            insight = llm_result['candidates'][0]['content']['parts'][0]['text']
            app.logger.info(f"LLM Insight: {insight}")
            return jsonify(insight=insight)
        else:
            app.logger.warning("Gemini API returned no candidates or empty content.")
            return jsonify(insight="Could not generate detailed insights at this time. Please try again."), 200

    except requests.exceptions.RequestException as req_err:
        app.logger.error(f"Error calling Gemini API: {req_err}", exc_info=True)
        return jsonify({'error': f'Failed to get insights from AI: {req_err}'}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during LLM insight generation: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while generating insights. Please try again later.'}), 500


if __name__ == '__main__':
    app.run(debug=True)