import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

# --- Custom CSS เพื่อความสวยงาม ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #2c3e50; text-align: center; font-weight: 800; }
    .stButton>button {
        background-color: #e74c3c; color: white; border-radius: 25px;
        padding: 10px 30px; border: none; font-weight: bold;
        font-size: 16px; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #c0392b; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 30px; border-radius: 15px; text-align: center;
        font-size: 24px; font-weight: bold; margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .risk { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    </style>
""", unsafe_allow_html=True)

# --- โหลดโมเดลและ scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ heart_disease_model.pkl หรือ scaler.pkl")
    st.info("กรุณาวางไฟล์ทั้งสองในโฟลเดอร์เดียวกับ app.py")
    st.stop()

# --- ส่วนหัวของเว็บไซต์ ---
st.markdown("<h1>🫀 ระบบประเมินความเสี่ยงโรคหัวใจ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>กรอกข้อมูลสุขภาพของคุณด้านล่าง เพื่อทำนายความเสี่ยงด้วย AI</p>", unsafe_allow_html=True)
st.markdown("---")

# --- ส่วนรับค่า Input ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📋 ข้อมูลทั่วไปและผลเลือด")
    age = st.slider("อายุ (ปี)", 20, 100, 50)
    sex = st.selectbox("เพศ", ["ชาย", "หญิง"])
    sex_val = 1 if sex == "ชาย" else 0
    
    cp = st.selectbox("ประเภทอาการเจ็บหน้าอก", 
                      ["1: Typical Angina", "2: Atypical Angina", "3: Non-anginal", "4: Asymptomatic"])
    cp_val = int(cp.split(':')[0])
    
    trestbps = st.slider("ความดันโลหิตขณะพัก (mm Hg)", 80, 200, 120)
    chol = st.slider("ระดับคอเลสเตอรอล (mg/dl)", 0, 600, 200)
    
    fbs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl?", ["ไม่ (0)", "ใช่ (1)"])
    fbs_val = 1 if "ใช่" in fbs else 0

with col2:
    st.markdown("##### 📋 ผลการตรวจคลื่นหัวใจและการออกกำลังกาย")
    restecg = st.selectbox("ผลคลื่นไฟฟ้าหัวใจขณะพัก", 
                           ["0: Normal", "1: ST-T abnormality", "2: LVH"])
    restecg_val = int(restecg.split(':')[0])
    
    thalach = st.slider("อัตราการเต้นของหัวใจสูงสุด (bpm)", 60, 220, 150)
    
    exang = st.selectbox("มีอาการเจ็บหน้าอกขณะออกกำลังกาย?", ["ไม่ (0)", "ใช่ (1)"])
    exang_val = 1 if "ใช่" in exang else 0
    
    oldpeak = st.slider("ค่า ST depression", 0.0, 6.0, 1.0, 0.1)
    
    slope = st.selectbox("ความชันของ ST segment", 
                         ["1: Up", "2: Flat", "3: Down"])
    slope_val = int(slope.split(':')[0])

st.markdown("---")

# --- ปุ่มกดทำนายและแสดงผล ---
if st.button("🔍 ประเมินความเสี่ยงทันที"):
    # ⚠️ สำคัญมาก: ลำดับคอลัมน์ต้องตรงกับตอนเทรนโมเดลเป๊ะๆ
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_val],
        'ChestPainType': [cp_val],
        'RestingBP': [trestbps],
        'Cholesterol': [chol],
        'FastingBS': [fbs_val],
        'RestingECG': [restecg_val],
        'MaxHR': [thalach],
        'ExerciseAngina': [exang_val],
        'Oldpeak': [oldpeak],
        'ST_Slope': [slope_val]
    })

    # ✅ แก้ไขแล้ว: แปลงเป็น Numpy Array เพื่อข้ามการตรวจสอบ feature names
    input_array = input_data.values
    input_scaled = scaler.transform(input_array)

    # ทำนายผล
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # แสดงผลแบบสวยงาม
    if prediction == 0:
        st.markdown('<div class="result-box safe">✅ ผลลัพธ์: คุณมีความเสี่ยงโรคหัวใจต่ำ (ปกติ)<br><small>รักษาสุขภาพต่อไปนะครับ!</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box risk">⚠️ ผลลัพธ์: คุณมีความเสี่ยงโรคหัวใจสูง<br><small>แนะนำให้ปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อตรวจละเอียด</small></div>', unsafe_allow_html=True)

    # แสดงความมั่นใจของโมเดล
    st.markdown("### 📊 ความมั่นใจของการทำนาย")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("โอกาสเป็นโรคหัวใจ", f"{prediction_proba[1]*100:.2f}%")
    with col_b:
        st.metric("โอกาสไม่เป็นโรคหัวใจ", f"{prediction_proba[0]*100:.2f}%")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>© 2026 Heart Disease Predictor | Powered by Decision Tree + Streamlit</p>", unsafe_allow_html=True)