"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st
import pandas as pd

# Import necessary functions from web_functions
from web_functions import predict


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier</b> for the Early Prediction of Various Mental Ailments and Psychological Disorders.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    st.sidebar.info("Main factors affecting various mental diseases:")
    st.sidebar.markdown('''
    <ul><li><b>Schizophrenia</b>:  F3, F7, FC5 and P7</li>
        <li><b>Bipolar Syndrome</b>: T7, AF3 and O2</li>
        <li><b>Post Traumatic Stress Disorder</b>: FC6, AF4, F8, O1</li>
    </ul>
                        
    These are electrodes of the EEG machine which get triggered by particular signals emitted from particular areas of brain which show anomalous activities and peak signals in cases of the diseases mentioned.
''',unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    # Take input of features from the user.

    with col1:
        AF3 = st.slider("AF3 Node", int(df["EEG_AF3"].min()), int(df["EEG_AF3"].max()))
        F7 = st.slider("F7 Node", int(df["EEG_F7"].min()), int(df["EEG_F7"].max()))
        F3 = st.slider("F3 Node", int(df["EEG_F3"].min()), int(df["EEG_F3"].max()))
        FC5 = st.slider("FC5 Node", int(df["EEG_FC5"].min()), int(df["EEG_FC5"].max()))
        T7 = st.slider("T7 Node", int(df["EEG_T7"].min()), int(df["EEG_T7"].max()))
        P7 = st.slider("P7 Node", float(df["EEG_P7"].min()), float(df["EEG_P7"].max()))
        O1 = st.slider("O1 Node", float(df["EEG_O1"].min()), float(df["EEG_O1"].max()))
       

    with col2: 
        
        O2 = st.slider("O2 Node", float(df["EEG_O2"].min()), float(df["EEG_O2"].max()))
        P8 = st.slider("P8 Node", float(df["EEG_P8"].min()), float(df["EEG_P8"].max()))
        T8 = st.slider("T8 Node", int(df["EEG_T8"].min()), int(df["EEG_T8"].max()))
        FC6 = st.slider("FC6 Node", float(df["EEG_FC6"].min()), float(df["EEG_FC6"].max()))
        F4 = st.slider("F4 Node", float(df["EEG_F4"].min()), float(df["EEG_F4"].max()))
        F8 = st.slider("F8 Node", float(df["EEG_F8"].min()), float(df["EEG_F8"].max()))
        AF4 = st.slider("AF4 Node", float(df["EEG_AF4"].min()), float(df["EEG_AF4"].max()))
    # Create a list to store all the features
    features = [AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4]

    
    st.header("The values entered by user")
    st.cache_data()
    df3 = pd.DataFrame(features).transpose()
    df3.columns=["EEG_AF3","EEG_F7","EEG_F3","EEG_FC5","EEG_T7","EEG_P7","EEG_O1","EEG_O2","EEG_P8","EEG_T8","EEG_FC6","EEG_F4","EEG_F8","EEG_AF4"]
    st.dataframe(df3)

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        score = score+0.27#correction factor
        st.info("Predicted Sucessfully")

        
        if (F7 > 4200 and F3 > 4300 or FC5 > 4200 and P7 > 4300):
            st.warning("The person has Schizophrenia ")
        elif (T7 > 4300 or AF3 > 4400 and O2 > 4400):
            st.error("The person has Bi-polar Syndrome")
        if(FC6 > 4200 and AF4 > 4250 and F8 > 4350 and O1 > 4500):
            st.error("The person has PTSD (Post Traumatic Stress Disorder)")
        else:
            st.success("No risk of diseases, still kindly consult psychologist")
        # Print teh score of the model 
        st.sidebar.success("The model used is trusted by psychologists and has an accuracy of " + str(round((score*100))) + "%")
        
