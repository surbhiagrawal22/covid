import numpy as np
import pandas as pd
import pickle
import joblib
import streamlit as st
from joblib import dump, load
import sklearn
import warnings
import shap
import requests
import datetime
import json
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
def predict_mental_health(classifier,data):
    prediction=classifier.predict(data)
    print(prediction)
    return prediction



def main():

    st.set_page_config(page_title='MentalHealthDetection', page_icon="🖖")
    st.title("Hello. We hope you are fine, Let's get started with the test")
    st.markdown('               __Get start with our App now at a finger click.__             ')
    st.sidebar.info('This application is created to predict Mental Health')

    # set background, use base64 to read local file
    def get_base64_of_bin_file(bin_file):
        """
        function to read png file 
        ----------
        bin_file: png -> the background image in local folder
        """
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_png_as_page_bg(png_file):
        """
        function to display png as bg
        ----------
        png_file: png -> the background image in local folder
        """
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        st.App {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return
    import base64
    set_png_as_page_bg("v870-tang-36.jpg")
    def nocall():
        main_bg = "v870-tang-36.jpg" #Hospital Image Background
        main_bg_ext = "jpg"

        side_bg =  "v870-tang-36.jpg"
        side_bg_ext = "jpg"
        st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

        st.markdown(f"""<style>.reportview-container {{background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})}}.sidebar .sidebar-content {{background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})}}</style>""",unsafe_allow_html=True)
    
   
    Classifer_name=["KNN","Random Forest"]
    st.sidebar.title('Mental Health Detection')
    st.sidebar.header('')
    st.sidebar.subheader("It's quick, easy and free")
  
    model_option = st.sidebar.selectbox(label='Select the Model',options=Classifer_name)
    'Selected Model :', model_option
    add_selectbox = st.sidebar.selectbox("How would you like to predict?",("Online", "Batch","About"))

    def select_mode(model_option):
        if model_option=="SVC":
            pickle_in = open("svc_SMOTE_tomek_classifier.pkl","rb")
            mclassifier=pickle.load(pickle_in)
            return mclassifier
        elif model_option=="KNN":
            pickle_in = open("knn_smote_classifier.pkl","rb")
            mclassifier=pickle.load(pickle_in)
            return mclassifier
        elif model_option=="Random Forest":     
            pickle_in = open("rfc_smote_tomek_classifier.pkl","rb")
            mclassifier=pickle.load(pickle_in)
            return mclassifier

    if add_selectbox == 'Online':

        # Coding Gender Column  
        Dem_Gender= st.selectbox(label="Dem Gender",options=["Male","Female"])
        if Dem_Gender=="Male":
            Dem_gender_Male=1
        else:
            Dem_gender_Male=0    

        # Coding Education Column  
        Dem_Education = st.selectbox("Dem Education",options=["- PhD/Doctorate","- Some College, short continuing education or equivalent",
                                                          "- Up to 12 years of school","- Up to 6 years of school","- Up to 9 years of school",
                                                          "- College degree, bachelor, master"])
        Dem_edu_PhD_Doctorate,	Dem_edu_Some_College_short,	Dem_edu_Upto12yearsofschool=0,0,0
        Dem_edu_Upto6yearsofschool,	Dem_edu_Upto9yearsofschool=0,0
        if Dem_Education=="- PhD/Doctorate":
            Dem_edu_PhD_Doctorate=1
        elif Dem_Education=="- Some College, short continuing education or equivalent":
            Dem_edu_Some_College_short=1
        elif Dem_Education=="- Up to 12 years of school":
            Dem_edu_Upto12yearsofschool=1
        elif Dem_Education=="- Up to 6 years of school":
            Dem_edu_Upto6yearsofschool=1
        elif Dem_Education=="- Up to 9 years of school":
            Dem_edu_Upto9yearsofschool=1
        elif Dem_Education=="- College degree, bachelor, master":
            pass
        
         # Coding Education of mother Column  
        Dem_Education_Mother = st.selectbox("Dem Education Mother",options=["- PhD/Doctorate", "- Some College or equivalent",
                                                          "- Up to 12 years of school","- Up to 6 years of school","- Up to 9 years of school",
                                                          "- College degree"])

        Dem_edu_mom_PhD_Doctorate,Dem_edu_mom_SomeCollegeorequivalent, Dem_edu_mom_Upto12yearsofschool ,Dem_edu_mom_Upto6yearsofschool,	Dem_edu_mom_Upto9yearsofschool=0,0,0,0,0                                              
        if Dem_Education_Mother =="- PhD/Doctorate":
            Dem_edu_mom_PhD_Doctorate=1
        elif Dem_Education_Mother =="- Some College or equivalent":   
            Dem_edu_mom_SomeCollegeorequivalent=1
        elif Dem_Education_Mother =="- Up to 12 years of school":   
            Dem_edu_mom_Upto12yearsofschool=1    
        elif Dem_Education_Mother =="- Up to 6 years of school":   
            Dem_edu_mom_Upto6yearsofschool=1  
        elif Dem_Education_Mother =="- Up to 9 years of school":   
            Dem_edu_mom_Upto9yearsofschool=1 
        elif Dem_Education_Mother=="- College degree":
            pass



        # Coding Employment Column                                                      
        Dem_Employment = st.selectbox("Dem Employment",["Not employed",	"Part time employed","Retired",
                                               	"Self-employed","Student","Full time employed"])

        Dem_employment_Not_employed,Dem_employment_Parttimeemployed,Dem_employment_Retired,Dem_employment_Selfemployed,Dem_employment_Student=0,0,0,0,0
        if Dem_Employment =="Not employed":
            Dem_employment_Not_employed=1
        elif Dem_Employment =="Part time employed":
            Dem_employment_Parttimeemployed=1
        elif Dem_Employment =="Retired":
            Dem_employment_Retired=1
        elif Dem_Employment =="Self-employed":
            Dem_employment_Selfemployed=1
        elif Dem_Employment =="Student":
            Dem_employment_Student=1
        elif Dem_Employment =="Full time employed":
            pass


        # Coding Expat Column                                           
        Dem_Expat = st.selectbox("Dem Expat",["yes","no"])
        Dem_Expat_yes=0
        if Dem_Expat =="yes":
            Dem_Expat_yes=1
        

        # Coding Marital Status Column
        Dem_maritalstatus=st.selectbox("Dem_maritalstatus",["Divorced/widowed",	"Married/cohabiting","Other or would rather not say",
                         	                          "Single"])
        Dem_maritalstatus_Married_cohabiting,	Dem_maritalstatus_Other,	Dem_maritalstatus_Single  =0,0,0                                          
        if Dem_maritalstatus=="Divorced/widowed":
            pass
        elif Dem_maritalstatus=="Married/cohabiting":  
            Dem_maritalstatus_Married_cohabiting=1
        elif Dem_maritalstatus=="Other or would rather not say":
            Dem_maritalstatus_Other=1
        elif  Dem_maritalstatus=="Single":
            Dem_maritalstatus_Single =1

       # Coding Riskgroup Column                                                    
        Dem_riskgroup=st.selectbox("Dem_riskgroup",["Not sure","Yes","No"])
        Dem_riskgroup_Not_sure,Dem_riskgroup_Yes=0,0
        if Dem_riskgroup=="Yes":
            Dem_riskgroup_Yes=1
        elif Dem_riskgroup=="Not sure":
            Dem_riskgroup_Not_sure=1
        if Dem_riskgroup=="No":
            pass


        Dem_age=st.number_input('Dem_age',min_value= 18, max_value=100)
        Dem_dependents=st.number_input("Dem_dependents")
        Dem_isolation_adults=st.number_input("Dem_isolation_adults")
        Dem_isolation_kids=st.number_input("Dem_isolation_kids")
        Trust_countrymeasure=st.number_input("Trust_countrymeasure")
        LONELINESS_Composite_Sum=st.number_input("LONELINESS_Composite_Sum")
        Neuroticism_Score_Sum=st.number_input("Neuroticism_Score_Sum")
        Extraversion_Score_Sum=st.number_input("Extraversion_Score_Sum")
        Openness_score_Sum=st.number_input("Openness_score_Sum")
        Agreeableness_Score_Sum=st.number_input("Agreeableness_Score_Sum")
        Conscientiousness_Score_Sum	=st.number_input("Conscientiousness_Score_Sum")
        Corona_concern_sum=st.number_input("Corona_concern_sum")
        OECD_insititutions_people_sum=st.number_input("OECD_insititutions_people_sum")
        SPS_sum=st.number_input("SPS_sum")
        Distress_sum=st.number_input("Distress_sum")
        
                     
        df=pd.DataFrame(data=[[Dem_gender_Male,
                         Dem_edu_PhD_Doctorate,	Dem_edu_Some_College_short,	Dem_edu_Upto12yearsofschool,Dem_edu_Upto6yearsofschool,	Dem_edu_Upto9yearsofschool,
                         Dem_edu_mom_PhD_Doctorate,Dem_edu_mom_SomeCollegeorequivalent, Dem_edu_mom_Upto12yearsofschool ,Dem_edu_mom_Upto6yearsofschool,	Dem_edu_mom_Upto9yearsofschool,
                         Dem_employment_Not_employed,Dem_employment_Parttimeemployed,Dem_employment_Retired,Dem_employment_Selfemployed,Dem_employment_Student,
                         Dem_Expat_yes,
                         Dem_maritalstatus_Married_cohabiting,	Dem_maritalstatus_Other,	Dem_maritalstatus_Single,
                         Dem_riskgroup_Not_sure,Dem_riskgroup_Yes,
                         Dem_age,Dem_dependents,Dem_isolation_adults,Dem_isolation_kids,Trust_countrymeasure,
                         LONELINESS_Composite_Sum,Neuroticism_Score_Sum,Extraversion_Score_Sum,Openness_score_Sum,
                         Agreeableness_Score_Sum,Conscientiousness_Score_Sum,Corona_concern_sum ,OECD_insititutions_people_sum,
                         SPS_sum,Distress_sum ]], columns=["Dem_gender_Male",
                         "Dem_edu_PhD_Doctorate",	"Dem_edu_Some_College_short",	"Dem_edu_Upto12yearsofschool","Dem_edu_Upto6yearsofschool",	"Dem_edu_Upto9yearsofschool",
                         "Dem_edu_mom_PhD_Doctorate","Dem_edu_mom_SomeCollegeorequivalent", "Dem_edu_mom_Upto12yearsofschool" ,"Dem_edu_mom_Upto6yearsofschool",	"Dem_edu_mom_Upto9yearsofschool",
                         "Dem_employment_Not_employed","Dem_employment_Parttimeemployed","Dem_employment_Retired","Dem_employment_Selfemployed","Dem_employment_Student",
                         "Dem_Expat_yes",
                         "Dem_maritalstatus_Married_cohabiting",	"Dem_maritalstatus_Other",	"Dem_maritalstatus_Single",
                         "Dem_riskgroup_Not_sure","Dem_riskgroup_Yes",
                         "Dem_age","Dem_dependents","Dem_isolation_adults","Dem_isolation_kids","Trust_countrymeasure",
                         "LONELINESS_Composite_Sum","Neuroticism_Score_Sum","Extraversion_Score_Sum","Openness_score_Sum",
                         "Agreeableness_Score_Sum","Conscientiousness_Score_Sum","Corona_concern_sum" ,"OECD_insititutions_people_sum",
                         "SPS_sum","Distress_sum" ])
        
        st.write("Original dataframe or User Input")
        st.write(df)    
        columns_to_scale = ['Dem_age', 'Dem_dependents', 'Dem_isolation_adults',
                            'Dem_isolation_kids', 'Trust_countrymeasure', 
                            'LONELINESS_Composite_Sum', 'Neuroticism_Score_Sum',
                            'Extraversion_Score_Sum', 'Openness_score_Sum',
                            'Agreeableness_Score_Sum', 'Conscientiousness_Score_Sum',
                            'Corona_concern_sum', 'OECD_insititutions_people_sum', 'SPS_sum',
                            'Distress_sum'
                            ]  

        # Load the scaler                          
        pickle_scalar = open("scaler1.pkl", "rb")
        scaler1=pickle.load(pickle_scalar)
        # Scale the numerical columns 
        df_scaledcolumns = pd.DataFrame(scaler1.transform(df[columns_to_scale]),columns=columns_to_scale) 
        # concat the scaled_columns_train dataframe and X_train dataframe
        df.drop(columns_to_scale, inplace=True, axis=1)
        df= pd.concat([df, df_scaledcolumns], axis=1)
        st.write("After scaling")
        st.write(df)

        # Makign the prediction on data 
        if st.button("Predict"):
            result=predict_mental_health(select_mode(model_option),df)
            output_dict = {1 : 'Highly Afected Mental- Needs Immediate Treatment', 0 : 'Low'}
        
            final_label = ""
            final_label = np.where(result == 1, 'Highly Afected Mental- Needs Immediate Treatment',np.where(result == 0,"Low","???????"))

            st.success(f'The output is {final_label}')    
   

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])   
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            st.write(data)
            predictions = predict_mental_health(select_mode(model_option),data)
            prediction=pd.DataFrame(predictions,columns=["Predicted Value"])
            prediction.loc[prediction["Predicted Value"]==1,"Predicted Value"]="Highly Affected Mental Health-Needs to be prioritised & Immediate Treatment"
            prediction.loc[prediction["Predicted Value"]==0,"Predicted Value"]="Low Affected Mental Health"
            actualvalue=pd.read_csv("y_smk_test.csv")
            st.write(prediction)
            st.write(actualvalue)
     
    if add_selectbox == 'About':
        st.subheader("Built with Streamlit and Sklearn")
        st.subheader("Surbhi Agrawal")
        st.subheader("https://github.com/surbhiagrawal22")

    
    st.button("Re-run")

if __name__=='__main__':
    main()
    

