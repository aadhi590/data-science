import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime

# App Configuration
st.set_page_config(page_title="Pharma Sales Prediction", page_icon="💊", layout="wide")
st.title(" Pharma Sales Prediction Dashboard")
st.markdown("Predict future drug category sales and analyze historical seasonality using Machine Learning.")
st.divider()

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('salesdaily.csv')
        if 'datum' in df.columns:
            df.rename(columns={'datum': 'Datum'}, inplace=True)
        # Preprocess 'Datum' column to extract Month and Day
        df['Datum'] = pd.to_datetime(df['Datum'])
        df['Month'] = df['Datum'].dt.month
        df['Day'] = df['Datum'].dt.day
        df['Year'] = df['Datum'].dt.year
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Identify drug categories by filtering out non-drug columns
    metadata_cols = ['Datum', 'Year', 'Month', 'Hour', 'Weekday Name', 'Day']
    drug_categories = [col for col in df.columns if col not in metadata_cols]

    # Sidebar UI
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004416.png", width=100) # Simple icon
    st.sidebar.header(" Prediction Settings")
    selected_category = st.sidebar.selectbox("Select Drug Category", drug_categories, help="Choose the ATC drug classification group.")
    
    # Future date selection
    max_date = df['Datum'].max().date()
    default_date = max_date + datetime.timedelta(days=1)
    
    st.sidebar.markdown("---")
    selected_date = st.sidebar.date_input("Select a Future Date", value=default_date)
    
    # Model Training
    @st.cache_resource
    def train_model(data, category):
        # Features & Target
        X = data[['Month', 'Day']]
        y = data[category]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return model, r2

    # --- TOP ROW: KPI Metrics ---
    # Calculate some stats for the selected category
    total_sales = df[selected_category].sum()
    avg_sales = df[selected_category].mean()
    max_sales_day = df.loc[df[selected_category].idxmax(), 'Datum'].date()
    max_sales_val = df[selected_category].max()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Historical Sales", f"{total_sales:,.0f}")
    kpi2.metric("Daily Average Sales", f"{avg_sales:.2f}")
    kpi3.metric("Record Sales Day", f"{max_sales_val:.0f}", str(max_sales_day), delta_color="off")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- MIDDLE ROW: CHARTS ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"####  Historical Timeline - **{selected_category}**")
        fig1 = px.line(df, x='Datum', y=selected_category, 
                       color_discrete_sequence=['#1f77b4'],
                       template='plotly_white')
        fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Date", yaxis_title="Quantity Sold")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown(f"#### 📊 Monthly Seasonality")
        monthly_avg = df.groupby('Month')[selected_category].mean().reset_index()
        fig2 = px.bar(monthly_avg, x='Month', y=selected_category,
                      color=selected_category, color_continuous_scale="Blues",
                      template='plotly_white')
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Month", yaxis_title="Avg Sales")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- BOTTOM ROW: MODELING & PREDICTIONS ---
    st.markdown("### 🔮 Future Forecast")
    
    # Train/Fetch model
    with st.spinner("Optimizing Random Forest..."):
        model, r2 = train_model(df, selected_category)
        
    future_month = selected_date.month
    future_day = selected_date.day
    
    pred_df = pd.DataFrame({'Month': [future_month], 'Day': [future_day]})
    prediction = model.predict(pred_df)[0]
    
    # Find historical average for THAT month to show delta
    hist_month_avg = df[df['Month'] == future_month][selected_category].mean()
    delta_val = prediction - hist_month_avg
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.info(" **Prediction Result**")
        st.metric(label=f"{selected_category} Estimated Sales for {selected_date}", 
                  value=f"{prediction:.2f}",
                  delta=f"{delta_val:.2f} vs Month Avg",
                  delta_color="normal")
                  
    with m_col2:
        st.success(" **Model Evaluation**")
        st.metric(label="R² Score (Accuracy Metric)", value=f"{r2:.3f}")
        st.caption("A higher R² indicates better predictive accuracy.")
        
    # Expander for Data
    with st.expander(" View Raw Dataset"):
        st.dataframe(df.sort_values('Datum', ascending=False).head(100), use_container_width=True)
    
else:
    st.warning(" Dataset 'salesdaily.csv' not found. Please ensure the Kaggle Pharma Sales dataset is downloaded and placed in the same directory as this script.")
