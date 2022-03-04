import streamlit as st 
import numpy as np 
import pandas as pd 
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




st.markdown("<h1 style='text-align: center; color: red;'>DIADETECT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>...a diabetes detection system</h4><br>", unsafe_allow_html=True)


st.write("Diabetes is a chronic disease that occurs when your blood glucose is too high. This application helps to effectively detect if someone has diabetes using Machine Learning and Python" )

# image = Image.open('C:/Users/CHIDERA ANI/Desktop/expert_system/db.jpg')
# st.sidebar.image(image, width=650)

#Get the data
df = pd.read_csv("C:/Users/CHIDERA ANI/Desktop/expert_system/diabetes.csv")


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
# replacing missing values
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)

df['Insulin'].fillna(df['Insulin'].median(), inplace = True)

df['BMI'].fillna(df['BMI'].median(), inplace = True)

X = df.drop(columns='Outcome')
y = df['Outcome']

#scaling
scaler = StandardScaler()
X =  pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Split the dataset into 75% Training set and 25% Testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)

name = st.text_input('What is your name?').upper()

#Get the feature input from the user
def get_user_input():

    age = st.number_input('Enter your age')
    pregnancies = st.number_input('How many times have you been pregnant?')
    glucose = st.number_input('What is your plasma glucose concentration?')
    blood_pressure = st.number_input('What is your blood pressure in mmHg?')
    skin_thickness = st.number_input('Enter your skin fold thickness in mm')
    insulin = st.number_input('Enter your insulin 2-Hour serum in mu U/ml')
    BMI = st.number_input('What is your Body Mass Index?')
    DPF = st.number_input('What is your Diabetes Pedigree Function?')
      
    
    user_data = {'Age': age,
                'Pregnancies': pregnancies,
              'Glucose': glucose,
                 'Blood Pressure': blood_pressure,
                 'Skin Thickness': skin_thickness,
                 'Insulin': insulin,
              'BMI': BMI,
              'DPF': DPF
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()


bt = st.button('Get Result')

if bt:
    rfmodel = RandomForestClassifier(random_state=1)
    rfmodel.fit(x_train, y_train)
    prediction = rfmodel.predict(user_input)
    # accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
    # acc = accuracy.format(round(accuracy*100), 2)


    if prediction == 1:
        st.write(name,", you either have diabetes or are likely to have it. Please visit the doctor as soon as possible.")
        st.write('The accuracy of this prediction is', str(accuracy_score(y_test, rfmodel.predict(x_test)) * 100) + '%' )
        
    else:
        st.write('Hurray!', name, 'You are diabetes FREE.')
        st.write('The accuracy of this prediction is',str(accuracy_score(y_test, rfmodel.predict(x_test)) * 100) + '%' )

