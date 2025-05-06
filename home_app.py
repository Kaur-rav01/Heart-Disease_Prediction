import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open(r'heart_pred_model.pkl','rb') as f:
    model=pickle.load(f)
with open(r'scaler.pkl','rb') as s:
    scaler=pickle.load(s)
    

        
st.set_page_config(page_title="Heart Disease Predictor",page_icon=":heart:",layout="wide")


    
st.sidebar.title("Navigation")
page=st.sidebar.radio("Go to",["🏠 Home", "📝 Prediction", "ℹ️ About"])



if page=="🏠 Home":
    st.markdown("<h1 style='text-align:center;color=#ff4b2b;'>Welcome to Heart Disease Predictor</h1>",unsafe_allow_html=True)
    st.write("")
    col1,col2,col3=st.columns([2,6,2])
    with col2:
        st.image(r'thi-heart-animation.webp',width=400)
    st.markdown(
    """
    <div style='text-align: center; font-size: 18px;'>
        <p>
        <strong>Heart disease</strong> is one of the leading causes of death in the world. In fact, in the United States, it is the leading cause of death, responsible for nearly 1 in every 5 deaths.
        </p>
        <p>
        It affects people of all ages and backgrounds, often developing silently over time due to lifestyle habits, medical conditions, or genetic factors.
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )
    
   
        
        

elif page=="📝 Prediction":
    st.markdown("""<h3 style='text-align:center;'>Heart Disease Risk Estimator</h3>""",unsafe_allow_html=True)
    st.write("Enter Patient's Information")
    
    age=st.select_slider("Age",options=range(0,101))
    
    col4,col5=st.columns(2)
    sex=col4.radio("Gender",["Male","Female"])
    sex=1 if sex=="Male" else 0
    
    chest=col5.selectbox("Chest Pain Type",['Typical Angina(0)', 'Atypical Angina(1)', 'Non-anginal Pain(2)', 'Asymptomatic(3)'])
    chest=int(chest.split('(')[-1][0])
    
    col6,col7=st.columns(2)
    with col6:
        trestbp=st.number_input("Resting Blood Pressure(mm/hg)",80,200,120)
        chol=st.number_input("Serum Cholesterol(mg/dl)",180,600,200)
        
    with col7:
        fbs=st.radio("Fasting Blood Sugar>120 (mg/dl)?",["Yes","No"])
        fbs=1 if fbs=="Yes" else 0
        restecg=st.selectbox("Resting ECG Results",['Normal(0)','ST-T wave-abnormality(1)','Left venticular hypertrophy(2)'])
        restecg=int(restecg.split('(')[-1][0])
    thalach=st.number_input("Maximum Heart Rate Achieved",60,220,150)
    
    col8,col9=st.columns(2)
    with col8:
        exang=st.radio("Exercise Induced Angina",["Yes","No"])
        exang=1 if exang=="Yes" else 0
        old_peak=st.number_input("OLD Peak(ST depression)",0.0,6.0,1.0)
        
    with col9:
        slope=st.selectbox("Slope of peak Exercise ST",["Unsloping(0)","Flat(1)","Downsloping(2)"])
        slope=int(slope.split('(')[-1][0])
        ca=st.selectbox("Number of major vessels covered with flourosopy(0-3)",[0,1,2,3])
    thal=st.selectbox("Thalssemia",["Normal(1)","Fixed Defect(2)","Reversible Defect(3)"])
    thal=int(thal.split('(')[-1][0])
    
    submit=st.button("Submit")
    if submit:
        input_data=np.array([age,sex,chest,trestbp,chol,fbs,restecg,thalach,exang,old_peak,slope,ca,thal])
        input_data_scaled=scaler.transform(input_data.reshape(1,-1))
        prediction=model.predict(input_data_scaled)
        
        
        if prediction[0]==1:
            st.error("⚠️ High risk of Heart Disease")
        else:
            st.success("✅ Low risk of Heart Disease")
            
            
elif page=="ℹ️ About":
    
    st.header('🔍 About It')
    st.write("""
    Heart disease remains one of the leading causes of death worldwide. Early detection and timely intervention can significantly improve health outcomes.
    This project uses Machine Learning to analyze health data and predict the likelihood of heart disease in individuals.
    """)



    st.header('🧠 How It Works')
    st.write("""
    Our model is trained on a dataset that includes attributes like:

    - Age
    - Gender
    - Blood Pressure
    - Cholesterol Levels
    - Chest Pain Type
    - ECG Results
    - Maximum Heart Rate
    - Exercise-Induced Angina

    With this information, the model predicts whether a person is at risk of heart disease.
    """)
    


    
