import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('tuned_random_forest_model.joblib')

# Load the scaler used
scaler = joblib.load('minmax_scaler.joblib')
# scaler = joblib.load('standard_scaler.joblib')

# Create a Streamlit interface to take user input and make predictions
st.title('Flight Booking Prediction Model Test and Deployment')

# Add input elements for the encoded features using selectbox and number_input

# input_feature_gender = st.selectbox('Gender', ("Male", "Female"))

input_feature_num_passengers = st.number_input('Number of Passengers')
input_feature_purchase_lead = st.number_input('Purchase Lead Amount')
input_feature_length_of_stay = st.number_input('Length of Stay (Hour)')
input_feature_flight_hour = st.number_input('Flight Hour')
input_feature_route = st.number_input('Route')
input_feature_booking_origin = st.number_input('Booking Origin')
input_feature_wants_extra_baggage = st.number_input('Wants Extra Baggage')
input_feature_wants_preferred_seat = st.number_input('Wants Preferred Seats')
input_feature_wants_in_flight_meals = st.number_input('Wants In-Flight Meals')
input_feature_flight_duration = st.number_input('Flight Duration')
input_feature_sales_channel_Internet = st.number_input('Sales Channel Internet')
input_feature_sales_channel_Mobile = st.number_input('Sales Channel Mobile')
input_feature_trip_type_CircleTrip = st.number_input('Trip Type Circle Trip')
input_feature_trip_type_OneWay = st.number_input('Trip Type One Way Trip')
input_feature_trip_type_RoundTrip = st.number_input('Trip Type Round Trip')
input_feature_flight_day_Fri = st.number_input('Flight Day Friday')
input_feature_flight_day_Mon = st.number_input('Flight Day Monday')
input_feature_flight_day_Sat = st.number_input('Flight Day Saturday')
input_feature_flight_day_Sun = st.number_input('Flight Day Sunday')
input_feature_flight_day_Thu = st.number_input('Flight Day Thursday')
input_feature_flight_day_Tue = st.number_input('Flight Day Tuesday')
input_feature_flight_day_Wed = st.number_input('Flight Day Wednesday')

# When the user clicks the 'Predict' button, make predictions using the loaded model
if st.button('Predict'):
    # Organize the user inputs into a list or array
    user_inputs = [input_feature_num_passengers,
                   input_feature_purchase_lead,
                   input_feature_length_of_stay,
                   input_feature_flight_hour,
                   input_feature_route,
                   input_feature_booking_origin,
                   input_feature_wants_extra_baggage,
                   input_feature_wants_preferred_seat,
                   input_feature_wants_in_flight_meals,
                   input_feature_flight_duration,
                   input_feature_sales_channel_Internet,
                   input_feature_sales_channel_Mobile,
                   input_feature_trip_type_CircleTrip,
                   input_feature_trip_type_OneWay,
                   input_feature_trip_type_RoundTrip,
                   input_feature_flight_day_Fri,
                   input_feature_flight_day_Mon,
                   input_feature_flight_day_Sat,
                   input_feature_flight_day_Sun,
                   input_feature_flight_day_Thu,
                   input_feature_flight_day_Tue,
                   input_feature_flight_day_Wed
                   ]

    # Convert the user_inputs list to a numpy array
    user_inputs_array = np.array(user_inputs)

    # Reshape the array to make it 2D
    reshaped_inputs = user_inputs_array.reshape(1, -1)

    # # Scale the user inputs using the loaded scaler
    # scaled_inputs = scaler.transform(reshaped_inputs)

    # Make predictions
    prediction = model.predict(reshaped_inputs)

    # Display the prediction
    st.success(f'The predicted class is {prediction[0]} (0: Not potential to complete booking, 1: Potential to complete booking)')

