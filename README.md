---
title: British Airways Flight Booking Prediction
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

![Project Header](https://raw.githubusercontent.com/mcikalmerdeka/British-Airways-Flight-Booking-Prediction/refs/heads/main/assets/Project%20Header.jpg)

Try the app on [Hugging Face Space](https://huggingface.co/spaces/mcikalmerdeka/british-airways-booking-prediction)

## 📌 Problem Statement

British Airways faces challenges in customer acquisition, with traditional reactive approaches becoming less effective in today's digital marketplace. With customers having access to extensive information online, waiting for customers to make bookings at the airport is too late. The airline needs a proactive approach to identify and target potential customers before they make their travel decisions, requiring data-driven predictive capabilities to understand customer booking behavior in advance.

## 📌 Goals

- Enhanced customer acquisition: Develop machine learning models to predict customer booking likelihood, enabling proactive targeting of potential customers. (**MAIN**)
- Improved business intelligence: Understand factors influencing customer booking decisions to optimize marketing strategies. (**SECONDARY**)

## 📌 Objectives

The ultimate goal of this project is to create a machine learning model that can:

- Accurately predict which customers are likely to book flights or holidays
- Identify key variables that influence booking decisions
- Provide actionable insights to improve customer acquisition strategies before customers embark on their holidays

## 📌 Business Metrics

- Customer Acquisition Rate: The rate at which the airline acquires new customers who book flights or holidays. (**MAIN**)
- Model Predictive Power: Evaluation metrics (accuracy, precision, recall) demonstrating the model's ability to identify potential customers. (**MAIN**)

## 📌 Project Methodology

1. **Data Exploration and Preparation**
   - Explore the provided customer booking dataset to understand different columns and basic statistics
   - Prepare and clean the dataset for predictive modeling
   - Create new features to enhance model performance

2. **Machine Learning Model Development**
   - Train a machine learning model (e.g., RandomForest) to predict customer booking behavior
   - Select algorithms that provide interpretability of variable contributions to predictive power

3. **Model Evaluation and Insights**
   - Conduct cross-validation and generate appropriate evaluation metrics
   - Create visualizations to interpret variable contributions
   - Summarize findings in a concise, manager-ready presentation

## 📌 Expected Outcomes

- A high-quality predictive model for customer booking behavior
- Clear understanding of factors influencing customer decisions
- Actionable insights for proactive customer acquisition strategies
- Single-slide summary of findings for management presentation

## 🚀 Quick Start

### Web Interface

## Install dependencies

using uv

```
uv sync
uv add -r requirements.txt
```

using pip:

```
pip install -r requirements.txt
```

## Run the Gradio app

```bash
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

## 📊 Features Used

The model uses the following features for prediction:

- **Booking Origin**: Country where the booking was made
- **Route**: Flight route (origin-destination)
- **Flight Duration**: Duration of the flight in hours
- **Length of Stay**: Number of days at destination
- **Sales Channel**: Internet or Mobile booking
- **Purchase Lead**: Days between booking and flight
- **Trip Type**: RoundTrip, OneWay, or CircleTrip
- **Flight Preferences**: Extra baggage, preferred seat, in-flight meals
- **Number of Passengers**: Total passengers in the booking

## 📁 Project Structure

```

.
├── app.py # Gradio web interface (entry point)
├── requirements.txt # Python dependencies
├── notebook_fix.ipynb # Jupyter notebook for training/analysis
├── README.md # This file
├── models/ # Trained model artifacts
│ ├── random_forest_model.joblib
│ ├── encoders.joblib
│ └── scalers.joblib
├── assets/ # Static assets
│ └── Project Header.jpg
└── data/ # Data files

```

## 📝 How to Use

1. **Manual Entry**:
   - Fill in trip details (passengers, route, duration)
   - Enter booking information (channel, origin, lead time)
   - Set flight schedule and preferences
   - Click "Predict" to see the result

2. **Upload CSV**:
   - Prepare a CSV with the required columns
   - Upload and get batch predictions
   - Download the results

3. **Sample Data**:
   - Use random samples from the test dataset
   - Evaluate model performance
   - View confusion matrix and statistics

## 📄 Files Managed by Git LFS

- `models/*.joblib` - Trained model files
- `assets/*.jpg` - Project images
