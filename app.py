import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import lime
from lime.lime_tabular import LimeTabularExplainer
import pickle
import os

# Flask app initialization
app = Flask(__name__)

# Filepath for the saved pipeline model
MODEL_PATH = "credit_risk_pipeline_lime.pkl"

# Step 1: Gather Data
def gather_data():
    traditional_data = pd.DataFrame({
        'credit_score': [750, 620, 680, 720, 610],
        'income': [50000, 30000, 40000, 45000, 28000],
        'spending': [20000, 15000, 18000, 21000, 16000]
    })

    alternative_data = pd.DataFrame({
        'social_media_sentiment': [0.8, 0.4, 0.6, 0.7, 0.3],
        'geolocation_stability': [0.9, 0.5, 0.7, 0.8, 0.4],
        'utility_bill_reliability': [1, 0, 1, 1, 0],
        'purchase_to_income_ratio': [0.4, 0.5, 0.45, 0.47, 0.57]
    })

    labels = pd.Series([1, 0, 1, 1, 0], name='repaid')  # Target variable

    data = pd.concat([traditional_data, alternative_data, labels], axis=1)
    return data

# Step 2: Train Model with Pipeline and LIME
def train_model_with_pipeline(data):
    X = data.drop(columns=['repaid'])
    y = data['repaid']

    # Create a pipeline with preprocessing and model training
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('model', RandomForestClassifier(random_state=42))  # Classifier
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the pipeline
    y_pred = pipeline.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save pipeline and LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X.columns.tolist(),
        class_names=["High Risk", "Low Risk"],
        mode="classification"
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "explainer": explainer, "X_test": X_test}, f)

    return pipeline

# Step 3: Real-Time API Endpoint
@app.route('/assess_credit', methods=['POST'])
def assess_credit():
    # Ensure the model is loaded
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found. Train the model first."}), 500

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        pipeline_model = model_data['pipeline']
        explainer = model_data['explainer']

    # Parse input JSON
    data = request.json
    input_data = pd.DataFrame([{
        "credit_score": data.get("credit_score"),
        "income": data.get("income"),
        "spending": data.get("spending"),
        "social_media_sentiment": data.get("social_media_sentiment"),
        "geolocation_stability": data.get("geolocation_stability"),
        "utility_bill_reliability": data.get("utility_bill_reliability"),
        "purchase_to_income_ratio": data.get("purchase_to_income_ratio")
    }])

    # Predict credit risk
    prediction = pipeline_model.predict(input_data)[0]
    risk = "Low Risk" if prediction == 1 else "High Risk"

    # Explainability with LIME
    explanation = explainer.explain_instance(
        input_data.values[0],
        pipeline_model.predict_proba
    )
    explanation_text = explanation.as_list()

    # Return response
    return jsonify({
        "credit_risk": risk,
        "details": "Prediction made using a pipeline with traditional and alternative data.",
        "explanation": explanation_text
    })

if __name__ == "__main__":
    # Check if model exists, if not, train it
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        data = gather_data()
        train_model_with_pipeline(data)

    # Run the Flask app
    print("Starting the API...")
    app.run(debug=True)
