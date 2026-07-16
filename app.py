import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Score Analysis & Prediction", layout="wide")
sns.set_theme(style="darkgrid")

@st.cache_resource
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    
    df = pd.read_csv('exams.csv')
    
    df.rename(columns={
        'race/ethnicity': 'race_ethnicity',
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course',
        'math score': 'math_score',
        'reading score': 'reading_score',
        'writing_score': 'writing_score'
    }, inplace=True, errors='ignore')
    return model, df

model, df = load_assets()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Dashboard", "Exploratory Data Analysis", "Score Prediction"])

if page == "Dashboard":
    st.title("Student Performance Analytics")
    st.markdown("""
    ### Business Understanding
    This application identifies the key factors that influence students' examination scores. 
    Using this predictive model, educational institutions can identify students at risk of poor academic performance and provide timely interventions to improve their outcomes.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df))
    col2.metric("Average Math Score", round(df['math_score'].mean(), 2))
    col3.metric("Highest Correlation", "0.95 (Read vs Write)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Exploratory Data Analysis":
    st.title("Advanced Data Visualization")
    
    tab_dist, tab_corr, tab_cat = st.tabs(["Score Distribution", "Correlation Analysis", "Categorical Analysis"])

    with tab_dist:
        st.subheader("Mathematics Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['math_score'], kde=True, color='royalblue', ax=ax)
        plt.title("Distribution of Students' Mathematics Scores")
        st.pyplot(fig)
        st.info("Insight: Most students score between 60 and 80, indicating a normal score distribution.")

    with tab_corr:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        corr = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.write("Strong linear relationship between reading and writing scores supports mathematical performance.")

    with tab_cat:
        st.subheader("Score Comparison by Category")
        cat_feature = st.selectbox("Select Category:", 
                                  ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cat_feature, y='math_score', data=df, palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif page == "Score Prediction":
    st.title("Mathematics Score Prediction")
    st.write("Enter the student's details below to predict the mathematics score.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", df['gender'].unique())
            race = st.selectbox("Race/Ethnicity", df['race_ethnicity'].unique())
            parent = st.selectbox("Parental Education", df['parental_level_of_education'].unique())
            lunch = st.selectbox("Lunch Type", df['lunch'].unique())
        
        with c2:
            prep = st.selectbox("Test Prep Course", df['test_preparation_course'].unique())
            reading = st.slider("Reading Score", 0, 100, 70)
            writing = st.slider("Writing Score", 0, 100, 70)

        if st.button("Predict Score", use_container_width=True):
            input_df = pd.DataFrame([{
                'gender': gender,
                'race_ethnicity': race,
                'parental_level_of_education': parent,
                'lunch': lunch,
                'test_preparation_course': prep,
                'reading_score': reading,
                'writing_score': writing
            }])
            
            prediction = model.predict(input_df)[0]
            
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown(f"### Predicted Score: **{round(prediction, 2)}**")
                status = "PASSED" if prediction >= 60 else "REMEDIAL"
                st.markdown(f"Status: <span style='color:{'green' if status=='PASSED' else 'red'}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
            
            with col_res2:

                fig, ax = plt.subplots(figsize=(5, 1))
                plt.barh(['Math Score'], [prediction], color='teal')
                plt.xlim(0, 100)
                plt.axvline(60, color='red', linestyle='--')
                st.pyplot(fig)
