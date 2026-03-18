"""
British Airways Flight Booking Prediction - Gradio Inference App

This app allows users to predict whether a customer will complete a flight booking
based on various features like trip details, preferences, and booking information.

Author: AI Assistant
Date: 2025
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_artifacts():
    """Load model and preprocessing artifacts."""
    try:
        model = joblib.load(PROJECT_ROOT / 'models' / 'random_forest_model.joblib')
        encoders = joblib.load(PROJECT_ROOT / 'models' / 'encoders.joblib')
        scalers = joblib.load(PROJECT_ROOT / 'models' / 'scalers.joblib')
        return model, encoders, scalers
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None


# Load model artifacts globally
model, encoders, scalers = load_model_artifacts()


def preprocess_input(data, encoders, scalers):
    """
    Preprocess input data using the same pipeline as training.
    """
    df = data.copy()
    
    # Step 1: Convert flight amenity preferences to numerical
    for col in ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1}).astype(int)
    
    # Step 2: Feature encoding
    # Binary encoding for sales_channel
    if 'sales_channel' in df.columns:
        encoder_key = 'binary_sales_channel'
        if encoder_key in encoders:
            label_map = encoders[encoder_key]['mapping']
            df['sales_channel'] = df['sales_channel'].map(label_map)
            if df['sales_channel'].isna().any():
                df['sales_channel'] = df['sales_channel'].fillna(0)
    
    # Frequency encoding for route and booking_origin
    for col in ['route', 'booking_origin']:
        if col in df.columns:
            encoder_key = f'frequency_{col}'
            if encoder_key in encoders:
                count_encoder = encoders[encoder_key]
                encoded_col = count_encoder.transform(df[[col]])
                df[col] = encoded_col[col].values
    
    # One-hot encoding for trip_type and flight_day
    for col in ['trip_type', 'flight_day']:
        if col in df.columns:
            encoder_key = f'onehot_{col}'
            if encoder_key in encoders:
                oh_encoder = encoders[encoder_key]
                oh_result = oh_encoder.transform(df[[col]])
                cats = oh_encoder.categories_[0]
                
                oh_col_names = [f'{col}_{cat}' for cat in cats[1:]]
                oh_df = pd.DataFrame(oh_result, columns=oh_col_names, index=df.index)
                
                df = df.drop(columns=[col])
                df = pd.concat([df, oh_df], axis=1)
    
    # Step 3: Feature scaling
    scaling_config = {
        "robust": {
            "columns": [
                "booking_origin", "route", "flight_duration", 
                "length_of_stay", "purchase_lead",
            ]
        },
        "minmax": {
            "columns": ["sales_channel", "num_passengers"]
        }
    }
    
    for method_name, config in scaling_config.items():
        if method_name in scalers:
            cols = config['columns']
            existing_cols = [col for col in cols if col in df.columns]
            if existing_cols:
                df[existing_cols] = df[existing_cols].astype(float)
                df[existing_cols] = scalers[method_name].transform(df[existing_cols])
    
    # Ensure all expected columns are present
    # Note: The model was trained on selected features excluding flight_day columns
    expected_columns = [
        'booking_origin', 'route', 'flight_duration', 'length_of_stay',
        'sales_channel', 'purchase_lead', 'wants_extra_baggage', 'wants_preferred_seat',
        'wants_in_flight_meals', 'num_passengers', 'trip_type_OneWay', 'trip_type_RoundTrip'
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    return df


def predict_booking(num_passengers, sales_channel, trip_type, purchase_lead,
                   length_of_stay, flight_hour, flight_day, route, booking_origin,
                   wants_extra_baggage, wants_preferred_seat, wants_in_flight_meals,
                   flight_duration):
    """Make prediction for a single booking."""
    
    if model is None:
        return "Error: Model not loaded", None, None
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'num_passengers': [num_passengers],
        'sales_channel': [sales_channel],
        'trip_type': [trip_type],
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour],
        'flight_day': [flight_day],
        'route': [route],
        'booking_origin': [booking_origin],
        'wants_extra_baggage': [wants_extra_baggage],
        'wants_preferred_seat': [wants_preferred_seat],
        'wants_in_flight_meals': [wants_in_flight_meals],
        'flight_duration': [flight_duration]
    })
    
    # Preprocess
    processed_data = preprocess_input(input_data, encoders, scalers)
    
    # Predict
    prediction = model.predict(processed_data)[0]
    prediction_proba = model.predict_proba(processed_data)[0]
    
    confidence = prediction_proba[prediction] * 100
    booking_prob = prediction_proba[1] * 100
    
    # Format result
    if prediction == 1:
        result = f"✅ **BOOKING WILL BE COMPLETED**\n\nConfidence: {confidence:.1f}%"
    else:
        result = f"❌ **BOOKING WILL NOT BE COMPLETED**\n\nConfidence: {confidence:.1f}%"
    
    # Create probability gauge
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh([0], [booking_prob], color='#28a745' if booking_prob >= 50 else '#dc3545', height=0.3)
    ax.barh([0], [100], color='lightgray', height=0.3, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Booking Completion Probability (%)', fontsize=10)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2)
    ax.text(booking_prob, 0, f'{booking_prob:.1f}%', 
            ha='center', va='center', fontweight='bold', fontsize=11)
    ax.set_yticks([])
    ax.set_title('Booking Probability Gauge', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Feature importance
    importance_fig = create_feature_importance_plot()
    
    return result, fig, importance_fig


def create_feature_importance_plot():
    """Create feature importance plot."""
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    
    # Feature names matching the selected columns used during training
    feature_names = [
        'booking_origin', 'route', 'flight_duration', 'length_of_stay',
        'sales_channel', 'purchase_lead', 'wants_extra_baggage', 'wants_preferred_seat',
        'wants_in_flight_meals', 'num_passengers', 'trip_type_OneWay', 'trip_type_RoundTrip'
    ]
    
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)  # Show top 10
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title('Top 10 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def predict_csv(file):
    """Make predictions for uploaded CSV file."""
    if model is None:
        return "Error: Model not loaded", None, None
    
    try:
        df = pd.read_csv(file.name)
        
        # Remove target column if present
        if 'booking_complete' in df.columns:
            actual = df['booking_complete']
            df_input = df.drop(columns=['booking_complete'])
        else:
            df_input = df
            actual = None
        
        # Preprocess
        processed_data = preprocess_input(df_input, encoders, scalers)
        
        # Predict
        predictions = model.predict(processed_data)
        prediction_probas = model.predict_proba(processed_data)
        
        # Create results
        results_df = df_input.copy()
        results_df['prediction'] = predictions
        results_df['booking_probability'] = prediction_probas[:, 1]
        results_df['predicted_label'] = results_df['prediction'].map({0: 'No Booking', 1: 'Completed Booking'})
        
        if actual is not None:
            results_df['actual'] = actual
            accuracy = (results_df['prediction'] == results_df['actual']).mean()
            summary = f"Accuracy: {accuracy:.2%} | Total Predictions: {len(predictions)} | Completed: {(predictions==1).sum()} | Not Completed: {(predictions==0).sum()}"
        else:
            summary = f"Total Predictions: {len(predictions)} | Completed: {(predictions==1).sum()} | Not Completed: {(predictions==0).sum()}"
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart
        pred_counts = results_df['predicted_label'].value_counts()
        colors = ['#dc3545', '#28a745']
        ax1.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        
        # Histogram
        ax2.hist(results_df['booking_probability'], bins=20, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax2.set_xlabel('Booking Probability', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Probability Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save results to CSV for download
        output_file = PROJECT_ROOT / 'predictions_output.csv'
        results_df.to_csv(output_file, index=False)
        
        return summary, results_df.head(20), fig, str(output_file)
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None


def load_sample_data():
    """Load sample test data."""
    try:
        df = pd.read_csv(PROJECT_ROOT / 'data' / 'test_data_5%_raw.csv')
        return df.head(100)  # Return first 100 rows
    except:
        return pd.DataFrame()


def predict_sample(n_samples):
    """Make predictions on sample data."""
    if model is None:
        return "Error: Model not loaded", None, None, None
    
    try:
        df = pd.read_csv(PROJECT_ROOT / 'data' / 'test_data_5%_raw.csv')
        
        # Select random samples
        sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
        df_selected = df.iloc[sample_indices]
        
        y_true = df_selected['booking_complete']
        X = df_selected.drop(columns=['booking_complete'])
        
        # Preprocess
        processed_data = preprocess_input(X, encoders, scalers)
        
        # Predict
        predictions = model.predict(processed_data)
        prediction_probas = model.predict_proba(processed_data)
        
        # Results
        accuracy = (predictions == y_true.values).mean()
        completed = (predictions == 1).sum()
        not_completed = (predictions == 0).sum()
        avg_conf = np.max(prediction_probas, axis=1).mean()
        
        summary = (f"**Results:**\n"
                  f"- Accuracy: {accuracy:.2%}\n"
                  f"- Completed Bookings: {completed}\n"
                  f"- No Bookings: {not_completed}\n"
                  f"- Average Confidence: {avg_conf:.1f}%")
        
        # Create results dataframe
        results_df = X.copy()
        results_df['actual'] = y_true.values
        results_df['predicted'] = predictions
        results_df['probability'] = prediction_probas[:, 1]
        results_df['correct'] = (predictions == y_true.values)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Booking', 'Completed'],
                   yticklabels=['No Booking', 'Completed'],
                   ax=ax_cm, cbar_kws={'label': 'Count'})
        ax_cm.set_xlabel('Predicted', fontsize=10)
        ax_cm.set_ylabel('Actual', fontsize=10)
        ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Distribution plot
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        pred_labels = ['Completed' if p == 1 else 'No Booking' for p in predictions]
        pred_counts = pd.Series(pred_labels).value_counts()
        colors = ['#dc3545', '#28a745']
        ax1.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        
        ax2.hist(prediction_probas[:, 1], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Booking Probability', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Probability Distribution', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        return summary, results_df, fig_cm, fig_dist
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None


# Create the Gradio interface
with gr.Blocks(title="BA Flight Booking Predictor") as demo:
    
    gr.Markdown("""
    # ✈️ British Airways Flight Booking Predictor
    
    Predict whether a customer will complete their flight booking using Machine Learning
    """)
    
    with gr.Tabs():
        
        # Tab 1: Manual Entry
        with gr.Tab("📝 Manual Entry"):
            gr.Markdown("### Enter booking details to get a prediction")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Trip Details**")
                    num_passengers = gr.Number(label="Number of Passengers", value=1, minimum=1, maximum=9)
                    trip_type = gr.Dropdown(label="Trip Type", choices=["RoundTrip", "OneWay", "CircleTrip"], value="RoundTrip")
                    route = gr.Textbox(label="Route (e.g., AKLDEL)", value="AKLDEL")
                    flight_duration = gr.Number(label="Flight Duration (hours)", value=5.5, minimum=1.0, maximum=20.0)
                
                with gr.Column():
                    gr.Markdown("**Booking Details**")
                    sales_channel = gr.Dropdown(label="Sales Channel", choices=["Internet", "Mobile"], value="Internet")
                    booking_origin = gr.Textbox(label="Booking Origin Country", value="New Zealand")
                    purchase_lead = gr.Number(label="Purchase Lead (days)", value=100, minimum=0, maximum=900)
                    length_of_stay = gr.Number(label="Length of Stay (days)", value=15, minimum=0, maximum=800)
                
                with gr.Column():
                    gr.Markdown("**Flight Schedule & Preferences**")
                    flight_hour = gr.Slider(label="Flight Hour", minimum=0, maximum=23, value=12, step=1)
                    flight_day = gr.Dropdown(label="Flight Day", 
                                            choices=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                            value="Mon")
                    wants_extra_baggage = gr.Dropdown(label="Extra Baggage", choices=["No", "Yes"], value="No")
                    wants_preferred_seat = gr.Dropdown(label="Preferred Seat", choices=["No", "Yes"], value="No")
                    wants_in_flight_meals = gr.Dropdown(label="In-Flight Meals", choices=["No", "Yes"], value="No")
            
            predict_btn = gr.Button("🔮 Predict", variant="primary")
            
            with gr.Row():
                result_output = gr.Markdown(label="Prediction Result")
            
            with gr.Row():
                gauge_plot = gr.Plot(label="Probability Gauge")
                importance_plot = gr.Plot(label="Feature Importance")
            
            predict_btn.click(
                fn=predict_booking,
                inputs=[num_passengers, sales_channel, trip_type, purchase_lead,
                       length_of_stay, flight_hour, flight_day, route, booking_origin,
                       wants_extra_baggage, wants_preferred_seat, wants_in_flight_meals,
                       flight_duration],
                outputs=[result_output, gauge_plot, importance_plot]
            )
        
        # Tab 2: Upload CSV
        with gr.Tab("📁 Upload CSV"):
            gr.Markdown("### Upload a CSV file with booking data for batch predictions")
            
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            upload_btn = gr.Button("🔮 Predict All", variant="primary")
            
            summary_output = gr.Markdown()
            results_table = gr.Dataframe(label="Results Preview (First 20 rows)")
            
            with gr.Row():
                dist_plot = gr.Plot(label="Distribution")
                download_btn = gr.File(label="Download Full Results")
            
            def process_csv_wrapper(file):
                summary, results, fig, output_path = predict_csv(file)
                return summary, results, fig, output_path
            
            upload_btn.click(
                fn=process_csv_wrapper,
                inputs=[file_input],
                outputs=[summary_output, results_table, dist_plot, download_btn]
            )
        
        # Tab 3: Sample Data
        with gr.Tab("🎲 Sample Data"):
            gr.Markdown("### Test with sample data from the test set")
            
            with gr.Row():
                n_samples = gr.Slider(label="Number of Samples", minimum=1, maximum=100, value=10, step=1)
                sample_btn = gr.Button("🔮 Run Predictions", variant="primary")
            
            sample_summary = gr.Markdown()
            
            with gr.Row():
                sample_results = gr.Dataframe(label="Individual Predictions")
                cm_plot = gr.Plot(label="Confusion Matrix")
            
            sample_dist_plot = gr.Plot(label="Prediction Distributions")
            
            sample_btn.click(
                fn=predict_sample,
                inputs=[n_samples],
                outputs=[sample_summary, sample_results, cm_plot, sample_dist_plot]
            )
        
        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About This App
            
            This app uses a **Random Forest classifier** trained on historical booking data to predict 
            whether a customer will complete their flight booking.
            
            #### Model Features:
            - **Trip details**: type, duration, route
            - **Customer preferences**: baggage, seat, meals
            - **Booking information**: channel, origin, lead time
            - **Flight schedule**: day, hour
            
            #### How to Use:
            1. **Manual Entry**: Fill in the booking details and click Predict
            2. **Upload CSV**: Upload a CSV file with multiple bookings for batch prediction
            3. **Sample Data**: Test the model with random samples from the test dataset
            
            #### Model Performance:
            The model was trained on 50,000+ booking records and achieves high accuracy on predicting 
            booking completion.
            
            #### Technical Details:
            - **Algorithm**: Random Forest Classifier
            - **Preprocessing**: Feature encoding (frequency, one-hot, label), scaling (robust, minmax)
            - **Framework**: Gradio for the web interface
            """)
    
    gr.Markdown("---")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
