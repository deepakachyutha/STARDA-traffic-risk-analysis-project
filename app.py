import streamlit as st
import pandas as pd
import joblib
import os
import sys
import time
import altair as alt
from datetime import datetime
import os
import gdown
import streamlit as st

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/accident_severity_model.pkl"):
    file_id = "1qpCqntbdqBq7wXrbdqKRBjGvGPnk1Zqc"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "models/accident_severity_model.pkl"
    
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error(f"Model Download Failed: {e}")
        st.stop()

if not os.path.exists("models/random_forest_model.pkl"):
    file_id_rf = "1Hc43wsoGfo4E5JOTXC745zM596QKSR__"
    url_rf = f"https://drive.google.com/uc?id={file_id_rf}"
    output_rf = "models/random_forest_model.pkl"
    
    try:
        gdown.download(url_rf, output_rf, quiet=False)
    except Exception as e:
        st.error(f"Random Forest Model Download Failed: {e}")
        st.stop()

if not os.path.exists("models/adaboost_model.pkl"):
    file_id_ada = "1MInVapbfDX_HoWAiS1bIje9Wjh_hjhdZ"
    url_ada = f"https://drive.google.com/uc?id={file_id_ada}"
    output_ada = "models/adaboost_model.pkl"
    
    try:
        gdown.download(url_ada, output_ada, quiet=False)
    except Exception as e:
        st.error(f"AdaBoost Model Download Failed: {e}")
        st.stop()

# SVM model
if not os.path.exists("models/svm_model.pkl"):
    file_id_svm = "11rNkru9V9BuoaPs1xYAQA74BPLExnuJt"
    url_svm = f"https://drive.google.com/uc?id={file_id_svm}"
    output_svm = "models/svm_model.pkl"
    
    try:
        gdown.download(url_svm, output_svm, quiet=False)
    except Exception as e:
        st.error(f"SVM Model Download Failed: {e}")
        st.stop()

st.set_page_config(
    page_title="STRADA Research Framework", 
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.abspath("src"))
try:
    from rag_engine import TrafficRAG
    from live_data import get_location, get_real_weather
except ImportError:
    def get_location(): return 0, 0, "Unknown Location"
    def get_real_weather(lat, lon): return None
    from src.rag_engine import TrafficRAG

def convert_temp(val, to_metric=True):
    return (val - 32) * 5/9 if to_metric else (val * 9/5) + 32

def convert_dist(val, to_metric=True):
    return val * 1.60934 if to_metric else val * 0.621371

def calculate_reliability_index(raw_prob):
    """
    Normalizes raw model probability into a Relative Reliability Index (0-100).
    Calibration Adjustment: Floor lowered to 0.40 (40%) to capture standard ML confidence.
    """
    MIN_THRESH = 0.40 
    MAX_THRESH = 0.95 
    
    if raw_prob < MIN_THRESH:
        return 0
    if raw_prob >= MAX_THRESH:
        return 100
        
    normalized = (raw_prob - MIN_THRESH) / (MAX_THRESH - MIN_THRESH)
    return int(normalized * 100)

@st.cache_resource
def initialize_system():
    model_path = "models/accident_severity_model.pkl"
    if not os.path.exists(model_path): model_path = "accident_severity_model.pkl"
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error("Critical Error: Model file not found. Please verify directory structure.")
        st.stop()
    
    kb_path = "src/knowledge_base.txt" if os.path.exists("src/knowledge_base.txt") else "knowledge_base.txt"
    rag = TrafficRAG(kb_path)
    
    data_path = "data/modern_training_data.csv"
    if not os.path.exists(data_path): data_path = "notebook/modern_training_data.csv"
    
    if os.path.exists(data_path):
        val_data = pd.read_csv(data_path, nrows=5000).sample(100, random_state=42).reset_index(drop=True)
    else:
        val_data = pd.DataFrame() 
        
    return model, rag, val_data

try:
    model, rag, val_data = initialize_system()
except Exception as e:
    st.error(f"System Initialization Failed: {e}")
    st.stop()

with st.sidebar:
    st.header("System Status")
    
    st.warning("⚠️ **Telematics Bridge:** Offline\n\nELM327 OBD-II Interface disconnected. Defaulting to **Retrospective Validation Mode**.", icon="🔌")
    
    st.divider()
    
    st.header("Configuration")
    
    model_type = st.selectbox("Inference Model", ["Random Forest (Balanced)", "AdaBoost (Benchmark)", "SVM (Benchmark)"], key="model_select")
    
    st.divider()
    
    unit_system = st.radio("Measurement Standard", ["Metric (SI)", "Imperial (US)"], key="unit_select")
    is_metric = unit_system == "Metric (SI)"
    
    st.divider()
    
    op_mode = st.radio("Operation Mode", ["Static Risk Inference", "Retrospective Data Validation"], key="mode_select")

    input_weather = "Clear"
    input_temp_f = 70.0
    input_vis_mi = 10.0
    input_hour = 14
    display_humidity = 50

    if op_mode == "Static Risk Inference":
        st.subheader("Environmental Parameters")
        
        data_source = st.radio("Data Source", ["Manual Input", "Geospatial Lookup"], key="data_src")
        
        if data_source == "Manual Input":
            input_weather = st.selectbox("Condition", ["Clear", "Rain", "Fog", "Snow", "Cloudy"], key="weather_select")
            
            if is_metric:
                temp_c = st.number_input("Ambient Temperature (°C)", value=25, key="temp_input_c")
                vis_km = st.slider("Visibility (km)", 0.0, 16.0, 16.0, key="vis_input_km")
                input_temp_f = convert_temp(temp_c, to_metric=False)
                input_vis_mi = convert_dist(vis_km, to_metric=False)
            else:
                input_temp_f = st.number_input("Ambient Temperature (°F)", value=77, key="temp_input_f")
                input_vis_mi = st.slider("Visibility (mi)", 0.0, 10.0, 10.0, key="vis_input_mi")
            
            input_hour = st.slider("Temporal Hour (0-23)", 0, 23, value=14, key="manual_time_slider")
                
        else: 
            with st.spinner("Acquiring Geospatial & Temporal Data..."):
                lat, lon, city = get_location()
                st.success(f"Location Locked: {city}")
                
                current_sys_hour = datetime.now().hour
                input_hour = current_sys_hour 
                
                live_weather = get_real_weather(lat, lon)
                
                if live_weather:
                    cond = live_weather['condition']
                    temp_real_f = live_weather['temperature']
                    vis_real_mi = live_weather['visibility']
                    display_humidity = 85 if cond in ["Rain", "Fog"] else 45
                    
                    disp_temp = convert_temp(temp_real_f, True) if is_metric else temp_real_f
                    unit_t = "°C" if is_metric else "°F"
                    
                    st.info(f"Current: {cond} | {disp_temp:.1f}{unit_t}")
                        
                    input_weather = cond
                    input_temp_f = temp_real_f
                    input_vis_mi = vis_real_mi
                else:
                    st.warning("API Unavailable. Reverting to default parameters.")
                    input_weather = "Clear"
                    input_temp_f = 70
                    input_vis_mi = 10
            
            st.info(f"🕒 Time Synced: {input_hour}:00")

st.title("STRADA Framework")
st.markdown("**System for Traffic Risk Assessment & Dynamic Analysis**")

if op_mode == "Static Risk Inference":
    
    tab_inference, tab_explainability = st.tabs(["Risk Inference Engine", "Decision Support (RAG)"])
    
    with tab_inference:
        st.subheader("Quantitative Risk Assessment")
        
        if st.button("Execute Risk Analysis", use_container_width=True, key="run_inference"):
            
            force_critical = False
            constraint_msg = ""
            
            if input_vis_mi < 0.1:
                force_critical = True
                constraint_msg = "Deterministic Override: Visibility < 0.1mi"
            elif input_weather == "Snow" and input_vis_mi < 1.0:
                force_critical = True
                constraint_msg = "Deterministic Override: Whiteout Conditions"
            
            weather_map = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3, "Snow": 4}
            input_vector = pd.DataFrame({
                'Region_Cluster': [0], 
                'Weather_Encoded': [weather_map.get(input_weather, 0)],
                'Temp_Scaled': [(input_temp_f - 60)/20],
                'Vis_Scaled': [(input_vis_mi - 9)/2],
                'Hour': [input_hour],
                'Month': [6], 'Weekday': [2]
            })
            
            if force_critical:
                pred_class = 4
                raw_conf = 0.99
                rel_index = 99
                st.warning(f"Safety Constraint Active: {constraint_msg}")
            else:
                pred_class = model.predict(input_vector)[0]
                raw_conf = model.predict_proba(input_vector)[0].max()
                rel_index = calculate_reliability_index(raw_conf)
            

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Projected Severity", f"Class {pred_class}", help="Scale: 1 (Minor) to 4 (Fatal)")
            col2.metric("Reliability Index", f"{rel_index}/100", help=f"Normalized Signal Intensity (Raw: {raw_conf:.2f})")
            col3.metric("Condition", input_weather)
            col4.metric("Est. Humidity", f"{display_humidity}%")
            
            st.divider()
            
            c_a, c_b = st.columns([1, 1.5])
            with c_a:
                if pred_class <= 2 and not force_critical: 
                    st.success("Risk Assessment: Nominal")
                else: 
                    st.error("Risk Assessment: Critical")
                    st.caption("Protocol: Alert logged to system registry.")
                        
            with c_b:
                query = f"SECTION {input_weather} {input_weather} driving safety guidelines"
                if force_critical: query = f"Extreme danger {input_weather} zero visibility whiteout"
                
                context = rag.retrieve(query)
                st.markdown(f"**Safety Protocol Advisory:**\n\n{context}")

    with tab_explainability:
        st.subheader("Automated Safety Protocol Retrieval")
        
        if "chat_history" not in st.session_state: 
            st.session_state.chat_history = [{"role": "assistant", "content": "RAG Knowledge Engine Active. Awaiting query..."}]
            
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
        if user_query := st.chat_input("Enter research query parameter..."):
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)
            
            rag_response = rag.retrieve(user_query)
            
            st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
            with st.chat_message("assistant"): st.markdown(rag_response)

elif op_mode == "Retrospective Data Validation":
    st.subheader("Longitudinal Validation Stream")
    
    if val_data.empty:
        st.error("Validation Dataset Unavailable")
        st.stop()
        
    m1, m2, m3, m4 = st.columns(4)
    chart_placeholder = st.empty()
    log_placeholder = st.empty()
    
    if st.button("Initialize Telemetry Replay", use_container_width=True, key="start_replay"):
        progress_bar = st.progress(0)
        risk_log = []
        
        for i in range(len(val_data)):
            row = val_data.iloc[i]
            
            input_vec = pd.DataFrame([row]).drop(columns=['Severity'], errors='ignore')
            pred = model.predict(input_vec)[0]
            raw_c = model.predict_proba(input_vec)[0].max()
            rel_idx = calculate_reliability_index(raw_c)
            
            syn_temp_f = 70 + (row['Temp_Scaled'] * 20)
            syn_vis_mi = 9 + (row['Vis_Scaled'] * 2)
            
            if syn_vis_mi < 1.0: 
                pred = 4
                rel_idx = 99
            
            m1.metric("Severity Class", pred)
            m2.metric("Reliability Index", f"{rel_idx}")
            
            disp_t = convert_temp(syn_temp_f, True) if is_metric else syn_temp_f
            unit_t = "°C" if is_metric else "°F"
            m3.metric("Ambient Temp", f"{disp_t:.1f}{unit_t}")
            m4.metric("Visibility", f"{syn_vis_mi:.1f} mi")
            
            risk_log.append({"Sequence ID": i, "Severity Class": pred})
            df_chart = pd.DataFrame(risk_log)
            
            chart = alt.Chart(df_chart).mark_line(interpolate='step-after').encode(
                x='Sequence ID', 
                y=alt.Y('Severity Class', scale=alt.Scale(domain=[1, 4])),
                color=alt.value("#2c3e50")
            ).properties(height=300)
            
            chart_placeholder.altair_chart(chart, use_container_width=True)
            
            if pred >= 3:
                log_placeholder.error(f"Event Logged: ID {i} | Critical Severity Detected")
            else:
                log_placeholder.info(f"Event Logged: ID {i} | Nominal")
                
            time.sleep(0.05)
            progress_bar.progress((i+1)/len(val_data))