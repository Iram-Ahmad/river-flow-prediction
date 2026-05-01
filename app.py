import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="River Flow System", layout="wide")

st.title("🌊 River Flow Monitoring & Prediction System")

# Load data
data = pd.read_excel("indus_tarbela_flow.xlsx")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data = data.sort_values(by='Date')

# Sidebar
st.sidebar.header("Controls")

# River Selector
river = st.sidebar.selectbox("Select River", ["Indus - Tarbela"])

days_ahead = st.sidebar.slider("Predict Days Ahead", 1, 7, 1)

# Dataset + Download
st.subheader("📂 Dataset")
st.write(data)

csv = data.to_csv(index=False).encode('utf-8')
st.download_button("Download Data", csv, "river_data.csv", "text/csv")

# Statistics
st.subheader("📊 Current Statistics")
col1, col2, col3 = st.columns(3)

col1.metric("Latest Flow", int(data['Inflow'].iloc[-1]))
col2.metric("Max Flow", int(data['Inflow'].max()))
col3.metric("Min Flow", int(data['Inflow'].min()))

# Graph
st.subheader("📈 River Flow Trend")
st.line_chart(data.set_index('Date')['Inflow'])

# ML Model
data['Days'] = np.arange(len(data))
X = data[['Days']]
y = data['Inflow']

model = LinearRegression()
model.fit(X, y)

# 🔥 Model Evaluation (NEW)
y_pred = model.predict(X)
error = mean_absolute_error(y, y_pred)
st.write(f"📉 Model Error (MAE): {error:.2f}")

# Prediction
future = pd.DataFrame([[len(data) + days_ahead]], columns=['Days'])
prediction = model.predict(future)

st.subheader("🤖 Prediction")
st.write(f"Predicted Flow after {days_ahead} day(s): {int(prediction[0])}")

# 🔥 Next 7 Days Prediction (NEW)
future_days = np.arange(len(data), len(data)+7).reshape(-1,1)
future_predictions = model.predict(future_days)

st.subheader("📅 Next 7 Days Prediction")
st.line_chart(future_predictions)

# Alert
st.subheader("⚠️ System Alert")
if prediction[0] < 40000:
    st.error("Low Flow - Irrigation Risk")
elif prediction[0] > 80000:
    st.error("High Flow - Flood Risk")
else:
    st.success("Normal Flow")

# 🔥 Risk Highlight (NEW)
if prediction[0] > data['Inflow'].mean():
    st.warning("⚠️ Flow above average")

# Simulation Button
if st.button("Simulate New Data"):
    st.write("Simulated new flow:", int(prediction[0] + np.random.randint(-2000, 2000)))

# Footer
st.markdown("---")
st.write("Developed for Final Year Project")