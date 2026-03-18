---
## title: British Airways Flight Booking Prediction
emoji: ✈️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# British Airways Flight Booking Prediction

Project Header

A machine learning solution to predict customer flight booking behavior, enabling British Airways to implement proactive customer acquisition strategies.

Try the live app on [Hugging Face Space](https://huggingface.co/spaces/mcikalmerdeka/british-airways-booking-prediction)

## Project Overview

End-to-end data science project that analyzes customer booking patterns, flight preferences, and trip characteristics to predict whether a customer will complete their flight booking. Includes comprehensive EDA, preprocessing pipelines with feature engineering, Random Forest classification model, and an interactive Gradio web interface for real-time booking predictions.

## Key Results

- **Model Algorithm**: Random Forest Classifier
- **Performance Metrics**:
  - Accuracy: 88.56% ± 0.37%
  - Precision: 89.49% ± 0.54%
  - Recall: 84.14% ± 0.56%
  - F1 Score: 86.73% ± 0.43%
  - ROC-AUC: 95.00% ± 0.19%
- **Dataset Size**: 50,000 booking records (49,281 after preprocessing)
- **Features Used**: 12 engineered features including booking origin, route, flight duration, and customer preferences

## Project Structure

```
├── app.py                      # Gradio web interface (entry point)
├── notebook_fix.ipynb          # EDA and model training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .github/workflows/          # GitHub Actions for HF deployment
├── models/                     # Trained model artifacts
│   ├── random_forest_model.joblib
│   ├── encoders.joblib
│   └── scalers.joblib
├── assets/                     # Project images and media
├── data/                       # Dataset files
│   ├── customer_booking.csv
│   ├── test_data_5%_preprocessed.csv
│   └── test_data_5%_raw.csv
└── utils/                      # Reusable preprocessing and ML functions
    ├── preprocessing.py
    ├── feature_selection.py
    ├── classification_evals_and_tuning.py
    ├── visualization.py
    ├── statistics.py
    └── ab_testing.py
```

## Quick Start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Installation

```bash
# Clone repository
git clone https://github.com/mcikalmerdeka/British-Airways-Flight-Booking-Prediction.git
cd British-Airways-Flight-Booking-Prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies (using pip)
pip install -r requirements.txt

# Or using uv (faster alternative)
uv sync
```

### Run the App

```bash
python app.py
```

Access the app at `http://localhost:7860`

## Features

- **Multiple Input Options**: Manual individual entry or batch CSV upload
- **Interactive Predictions**: Real-time booking completion probability with confidence scores
- **Visual Analytics**:
  - Probability gauge visualization
  - Feature importance plots
  - Confusion matrix for model evaluation
  - Prediction distribution charts
- **Batch Processing**: Efficient processing of multiple booking records
- **Sample Data Testing**: Built-in test cases using actual test dataset
- **Model Performance Display**: View comprehensive evaluation metrics

## Technical Stack

- Python 3.12+
- scikit-learn (Random Forest, preprocessing, model evaluation)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- Gradio (web application)
- joblib (model serialization)
- category-encoders (frequency encoding)
- imbalanced-learn (SMOTE for class balancing)

## Business Problem

British Airways faces challenges in customer acquisition, with traditional reactive approaches becoming less effective in today's digital marketplace. With customers having access to extensive information online, waiting for customers to make bookings at the airport is too late. The airline needs a proactive approach to identify and target potential customers before they make their travel decisions.

### Problem Statement

How can we develop a machine learning system that predicts customer booking behavior to enable proactive targeting of potential customers and optimize marketing strategies?

### Business Metrics

- **Customer Acquisition Rate** [MAIN]: The rate at which the airline acquires new customers who book flights or holidays
- **Model Predictive Power** [MAIN]: Evaluation metrics (accuracy, precision, recall, ROC-AUC) demonstrating the model's ability to identify potential customers

### Goals

- **Primary**: Enhanced customer acquisition through machine learning implementation with high recall to identify potential bookers
- **Secondary**: Improved business intelligence to understand factors influencing customer booking decisions

### Objectives

1. Build a machine learning model that accurately predicts which customers are likely to book flights or holidays
2. Identify key variables that influence booking decisions
3. Provide actionable insights to improve customer acquisition strategies before customers embark on their holidays

## Model Methodology

### Data Preprocessing Pipeline

1. **Data Cleaning**: Duplicate removal (719 duplicates found and removed)
2. **Feature Engineering**:

- Extracted origin and destination airport codes from route
- Converted binary preference flags to categorical (Yes/No)

3. **Feature Encoding**:

- Binary encoding for sales channel
- Frequency encoding for high-cardinality features (route, booking_origin)
- One-hot encoding for trip type

4. **Feature Scaling**:

- Robust scaling for numerical features with outliers
- MinMax scaling for bounded features

5. **Class Balancing**: SMOTE oversampling (4:5 ratio) to handle 85% class imbalance

### Feature Selection

Selected 12 features based on statistical analysis:

- **booking_origin**: Country where booking was made (18.2% importance)
- **route**: Flight route code (12.7% importance)
- **flight_duration**: Flight duration in hours (9.5% importance)
- **length_of_stay**: Days at destination
- **purchase_lead**: Days between booking and travel
- **sales_channel**: Internet or Mobile
- **num_passengers**: Number of passengers
- **wants_extra_baggage**: Preference flag
- **wants_preferred_seat**: Preference flag
- **wants_in_flight_meals**: Preference flag
- **trip_type_OneWay**: One-hot encoded
- **trip_type_RoundTrip**: One-hot encoded

### Model Training

- **Algorithm**: Random Forest Classifier
- **Validation Strategy**: Train/Validation/Test split (75%/15%/5%)
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Class Imbalance Handling**: SMOTE oversampling on training set only

## Try the Live App

Link to the deployed app: [British Airways Flight Booking Prediction](https://huggingface.co/spaces/mcikalmerdeka/british-airways-booking-prediction)

## How to Use

1. **Manual Entry**:

- Fill in trip details (passengers, route, duration)
- Enter booking information (channel, origin country, lead time)
- Set flight schedule and preferences (baggage, seat, meals)
- Click "Predict" to see booking probability with confidence gauge

2. **Upload CSV**:

- Prepare CSV with required columns matching the dataset format
- Upload for batch predictions
- Download results with predictions and probabilities

3. **Sample Data**:

- Test with random samples from actual test dataset
- View confusion matrix and performance metrics
- Compare predicted vs actual outcomes

## Data Dictionary

| Feature               | Description                                | Type        | Importance |
| --------------------- | ------------------------------------------ | ----------- | ---------- |
| booking_origin        | Country where booking was made             | Categorical | 18.2%      |
| route                 | Flight route (origin-destination)          | Categorical | 12.7%      |
| flight_duration       | Total flight duration in hours             | Numerical   | 9.5%       |
| length_of_stay        | Number of days at destination              | Numerical   | High       |
| purchase_lead         | Days between booking and travel            | Numerical   | Medium     |
| sales_channel         | Booking channel (Internet/Mobile)          | Binary      | Low        |
| trip_type             | Type of trip (RoundTrip/OneWay/CircleTrip) | Categorical | Low        |
| num_passengers        | Number of passengers                       | Numerical   | Low        |
| wants_extra_baggage   | Extra baggage preference                   | Binary      | Medium     |
| wants_preferred_seat  | Preferred seat preference                  | Binary      | Medium     |
| wants_in_flight_meals | In-flight meals preference                 | Binary      | Low        |

## Key Insights

1. **Booking Origin** is the strongest predictor (18.2% importance) - customers from certain countries have significantly different booking patterns
2. **Route** is the second most important feature (12.7%) - specific destinations drive booking behavior
3. **Flight Duration** affects booking likelihood (9.5%) - longer flights show different patterns
4. **Customer Preferences** (baggage, seat, meals) collectively contribute to predictions
5. Model achieves excellent ROC-AUC of 95%, indicating strong discriminative power

## Files Managed by Git LFS

- `models/*.joblib` - Trained model files
- `assets/*.jpg` - Project images

## Author

**Muhammad Cikal Merdeka** | Data Analyst/Data Scientist

- [GitHub](https://github.com/mcikalmerdeka)
- [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
- [Email](mailto:mcikalmerdeka@gmail.com)

---
