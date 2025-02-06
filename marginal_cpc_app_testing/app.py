import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Load model function
def load_model(model_load_path):
    with open(model_load_path, 'rb') as file:
        saved_data = pickle.load(file)
        best_model = saved_data['model']
        scaler = saved_data['scaler']
        y_scaler = saved_data['y_scaler']
        feature_names = saved_data['feature_names']
        last_trained_date = saved_data.get('last_trained_date', None)
    return best_model, scaler, y_scaler, feature_names, last_trained_date

# Prediction function
def predict_cpc(model, scaler, y_scaler, features):
    features_scaled = scaler.transform([features])
    predicted_cpc_scaled = model.predict(features_scaled)
    predicted_cpc = y_scaler.inverse_transform(predicted_cpc_scaled.reshape(-1, 1)).flatten()
    return predicted_cpc[0]

# Plot function
def plot_interactive(df, cpc_col, spend_col, clicks_col, roi_col, value_per_click):
    df['Predicted_Clicks'] = df[spend_col] / df[cpc_col]
    df['ROI'] = df['Predicted_Clicks'] * value_per_click - df[spend_col]

    fig = px.scatter(df, x=cpc_col, y=spend_col, trendline="ols", labels={cpc_col: 'CPC', spend_col: 'Spend'},
                     title='Spend x Predicted CPC with Regression Line')

    fig.add_trace(go.Scatter(
        x=df[cpc_col], y=df[spend_col], mode='markers', name='Spend vs CPC',
        marker=dict(color='slategray')
    ))

    fig.update_layout(
        title='Spend x Predicted CPC',
        xaxis_title='CPC',
        yaxis_title='Spend',
        template='plotly_white',
        autosize=False,
        width=1080,
        height=720,
    )

    st.plotly_chart(fig)

st.title("Predict CPC Based on Daily Spend")
st.write("Input daily spend amounts for a week to get the predicted CPC, predicted clicks and estimated impressions.")

model_load_path = './Models/best_model_no_shap3.pkl'
model, scaler, y_scaler, feature_names, last_trained_date = load_model(model_load_path)

uploaded_file = st.file_uploader("Upload your past data CSV", type=["csv"])

def lag_features_new_data(new_data, column_names, lagged_features=True, rolling_avg=True):
    if lagged_features:
        for column in column_names:
            new_data['lag_' + column] = new_data[column].shift(1)
    if rolling_avg:
        for column in column_names:
            new_data['rolling_7_' + column] = new_data[column].rolling(window=7).mean()
            new_data['rolling_14_' + column] = new_data[column].rolling(window=14).mean()
    return new_data

if uploaded_file:
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    header_row = None
    expected_columns = ['Day', 'Cost', 'Clicks', 'Impr.', 'Search impr. share']

    for i, line in enumerate(lines):
        if all(col in line for col in expected_columns):
            header_row = i
            break

    if header_row is not None:
        past_data = pd.read_csv(uploaded_file, header=header_row)
        st.write("Past data uploaded successfully!")
        past_data['Day'] = pd.to_datetime(past_data['Day'])
        past_data = past_data.rename(
            columns={'Day': 'DATE', 'Cost': 'SPEND', 'Impr.': 'IMPRESSIONS', 'Clicks': 'CLICKS',
                     'Search impr. share': 'SEARCH_IMPRESSION_SHARE'})
        filtered_cpc = past_data.sort_values('DATE').reset_index(drop=True)
        filtered_cpc.set_index('DATE', inplace=True)
        filtered_cpc['SEARCH_IMPRESSION_SHARE'] = filtered_cpc['SEARCH_IMPRESSION_SHARE'].str.replace('<', '')
        filtered_cpc['SEARCH_IMPRESSION_SHARE'] = filtered_cpc['SEARCH_IMPRESSION_SHARE'].str.replace('%', '').astype(float)

        for col in filtered_cpc.select_dtypes(include=['category', 'object', 'string']).columns:
            filtered_cpc[col] = filtered_cpc[col].str.replace('.', '').str.replace(',', '')
            if filtered_cpc[col].str.isdigit().all():
                filtered_cpc[col] = pd.to_numeric(filtered_cpc[col], errors='coerce')

        filtered_cpc[['IMPRESSIONS', 'CLICKS', 'SPEND', 'SEARCH_IMPRESSION_SHARE']] = filtered_cpc[
            ['IMPRESSIONS', 'CLICKS', 'SPEND', 'SEARCH_IMPRESSION_SHARE']].astype(float)

        agg_dict = {col: 'sum' for col in filtered_cpc.select_dtypes(include=['number']).columns if col != 'SEARCH_IMPRESSION_SHARE'}
        agg_dict['SEARCH_IMPRESSION_SHARE'] = 'sum'

        filtered_cpc_resampled = filtered_cpc.resample('D').agg(agg_dict).reset_index()

        filtered_cpc_resampled['CPC'] = filtered_cpc_resampled['SPEND'] / filtered_cpc_resampled['CLICKS']
        filtered_cpc_resampled['ELIGIBLE_IMPRESSIONS'] = filtered_cpc_resampled['IMPRESSIONS'] / filtered_cpc_resampled['SEARCH_IMPRESSION_SHARE']

        new_data = filtered_cpc_resampled

        if not new_data.empty:
            column_names = ['SPEND', 'CLICKS', 'IMPRESSIONS', 'ELIGIBLE_IMPRESSIONS']
            new_dataset = lag_features_new_data(new_data, column_names, lagged_features=True, rolling_avg=True)

            new_dataset = new_dataset.fillna(0)

            X_new = new_dataset[feature_names]
            y_new = new_dataset['CPC']

            X_new_scaled = scaler.transform(X_new)
            y_new_scaled = y_scaler.transform(y_new.values.reshape(-1, 1)).flatten()

            model.fit(X_new_scaled, y_new_scaled)

            last_trained_date = new_data['DATE'].max()

            save_model(model, scaler, y_scaler, feature_names, last_trained_date, model_load_path)
            st.write("Model retrained and updated successfully!")
        else:
            st.write("No new data to update the model.")
    else:
        st.write("Could not find the header row. Please ensure the CSV file has the correct columns.")

spend = [st.slider(f"Enter spend amount for day {day + 1}:", min_value=0, max_value=25000, value=5000, step=100) for day in range(7)]

ctr = st.number_input("Enter current average CTR:", min_value=0.001, max_value=10.0, value=0.5)
total_spend = st.number_input("Enter previous week's total spend:", min_value=0, max_value=200000, value=35000, step=1)
total_clicks = st.number_input("Enter previous week's total clicks:", min_value=0, max_value=200000, value=10000, step=1)

value_per_click = total_spend / total_clicks if total_clicks != 0 else 0

# Create an empty DataFrame with required columns
data = pd.DataFrame(columns=feature_names)

# Manually add features for each day, ensuring each day's prediction is independent
for day in range(7):
    day_data = pd.DataFrame({
        'CLICKS': [0],
        'IMPRESSIONS': [0],
        'SEARCH_IMPRESSION_SHARE': [0],
        'CPC': [0],
        'ELIGIBLE_IMPRESSIONS': [0],
        'SPEND': [spend[day]]
    })
    day_data = lag_features_new_data(day_data, ['SPEND', 'CLICKS', 'IMPRESSIONS', 'ELIGIBLE_IMPRESSIONS'])
    day_data.fillna(0, inplace=True)

    for feature in feature_names:
        if feature not in day_data.columns:
            day_data[feature] = 0

    day_data = day_data[feature_names]
    data = pd.concat([data, day_data], ignore_index=True)

predicted_cpcs = [predict_cpc(model, scaler, y_scaler, data.iloc[day].values) for day in range(7)]

df_plot = pd.DataFrame({
    'Spend': spend,
    'Predicted CPC': predicted_cpcs
})
df_plot['Predicted Clicks'] = df_plot['Spend'] / df_plot['Predicted CPC']
df_plot['Estimated Impressions'] = df_plot['Predicted Clicks'] / ctr

if st.button("Predict CPC"):
    try:
        plot_interactive(df_plot, "Predicted CPC", "Spend", 'Predicted Clicks', 'ROI', value_per_click)

        for day in range(len(predicted_cpcs)):
            st.markdown(f"### Day {day + 1}:")
            st.markdown(f"- **Predicted CPC:** ${predicted_cpcs[day]:.2f}")
            st.markdown(f"- **Predicted Clicks:** {df_plot['Predicted Clicks'][day]:.2f}")
            st.markdown(f"- **Estimated Impressions:** {df_plot['Estimated Impressions'][day]:,.2f}")
            st.markdown("---")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
