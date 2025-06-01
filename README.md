# Multi-Disease Prediction System with AI Insights

## Overview

This project is a full-stack web application designed to predict the risk of various diseases (Diabetes, Heart Disease, Lung Cancer) based on user-provided health parameters. It features an interactive user interface, real-time risk visualization, and leverages the Google Gemini API to provide personalized health insights and advice.

## Features

* **Multi-Disease Prediction:** Predicts risk for Diabetes, Heart Disease, and Lung Cancer.
* **Interactive Forms:** User-friendly forms for inputting health data.
* **Real-time Risk Visualization:** Displays prediction results as dynamic, animated bar charts showing the percentage risk.
* **AI-Powered Health Insights:** Integrates the Google Gemini API to generate personalized, non-medical health advice and next steps based on prediction outcomes and user input.
* **Dynamic Content Loading:** Utilizes `<iframe>` elements to load disease-specific forms seamlessly without full page reloads, enhancing user experience.
* **Responsive Design:** Adapts to various screen sizes for optimal viewing on desktop and mobile devices.
* **Clear Health Tips:** Provides general health tips relevant to the predicted risk status.

## Technologies Used

**Backend:**
* **Flask (Python):** Web framework for handling routes, serving HTML, and processing prediction requests.
* **Scikit-learn:** Used for loading and utilizing pre-trained Machine Learning models (`.pkl` files).
* **NumPy:** For numerical operations on input data.
* **`requests` library:** For making HTTP calls to the Google Gemini API.
* **`logging`:** For application logging and debugging.

**Frontend:**
* **HTML5:** Structure of the web pages.
* **CSS3:** Styling and animated visualizations.
* **JavaScript (ES6+):** For dynamic content loading, form handling, API calls, and chart animations.
* **Font Awesome:** For icons.

**AI/ML:**
* **Google Gemini API (gemini-2.0-flash):** Large Language Model for generating contextual health insights.
* **Pre-trained ML Models (.pkl):** Binary classification models for disease prediction.

## Project Structure
```bash
|__train_models.ipynb       #for training and storing models
|__data_preprocessing.ipynb  #for data processing
|
├── app.py                  # Flask backend application
├── requirements.txt        # Python dependencies
|__data/
|   |__datasets
|
|__data_preprocessed/
|    |__preprocessed datasets
|
├── model_files/            # Directory for pre-trained ML models
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── lung_model.pkl
├── templates/              # HTML templates for Flask
│   ├── index.html          # Main dashboard page
│   ├── diabetes.html       # Diabetes prediction form
│   ├── heart.html          # Heart disease prediction form
│   └── lung_cancer.html    # Lung cancer prediction form
└── static/                 # Static assets (CSS, JS, images - if any)
  └── css/
        └── css_files       # styleing each templates
  |___ js/
        |__js_files
  |___images/
        |__image_files
```
## Setup and Installation

Follow these steps to set up and run the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd multi-disease-predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Make sure your `requirements.txt` contains `Flask`, `scikit-learn`, `numpy`, `requests`). If not, create it with:
    ```
    Flask
    scikit-learn
    numpy
    requests
    ```
    (You might want to specify exact versions for production, e.g., `Flask==2.3.3`).

4.  **Place your pre-trained models:**
    Ensure you have your `diabetes_model.pkl`, `heart_model.pkl`, and `lung_model.pkl` files inside the `model_files/` directory. These are essential for the prediction functionality.

5.  **Google Gemini API Key (for local testing):**
    * For local development, you would typically need a Google Cloud Project and enable the Gemini API, then generate an API key.
    * **However, for deployment in environments like Google Cloud's Canvas, the API key is automatically injected.** In `app.py`, the `api_key = ""` line is intentionally left blank for this reason. If you're running strictly locally outside of such an environment and want the LLM feature to work, you would temporarily replace `api_key = ""` with `api_key = "YOUR_GEMINI_API_KEY_HERE"` (but remember to revert before pushing to a public repo!).

## Usage
0. **Datasets setup and train models**
    ```bash
    data_preprocessing.ipynb
    train_models.ipynb
    ```

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    You should see output indicating that the Flask development server is running, typically on `http://127.0.0.1:5000/`.

2.  **Open in browser:**
    Navigate to `http://127.0.0.1:5000/` in your web browser.
    ![Screenshot 2025-06-01 203735](https://github.com/user-attachments/assets/7de0dd79-2e68-48e4-959e-ace9d91c808c)


4.  **Interact with the app:**
    * Select a disease from the navigation bar (Diabetes, Heart Disease, Lung Cancer).
    * Fill in the required health parameters in the form.
    * Click "Predict Risk" to see the prediction result and an animated risk bar.
    * Click "✨ Get More Insights" to receive AI-generated advice.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## Author
Jakir Hussian.
