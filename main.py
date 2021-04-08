import streamlit as st 
import numpy as np 
import pandas as pd 
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# st.title('Diabetes Detection')
st.title("Diabetes Detection System")
# st.markdown("<h1 style='text-align: center; color: pink;'>Diabetes Detection System</h1>", unsafe_allow_html=True)


st.write("This application helps to detect if someone has diabetes using Machine Learning and Python" )

image = Image.open('C:/Users/CHIDERA ANI/Desktop/expert_system/db.jpg')
st.image(image,use_column_width=True)

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

name = st.text_input('Your Name').upper()

#Get the feature input from the user
def get_user_input():

    age = st.number_input('Your age')
    pregnancies = st.number_input('How many times have you been pregnant?')
    glucose = st.number_input('Your plasma glucose concentration')
    blood_pressure = st.number_input('Your blood pressure in mmHg')
    skin_thickness = st.number_input('Your skin fold thickness in mm')
    insulin = st.number_input('Your insulin 2-Hour serum in mu U/ml')
    BMI = st.number_input('Your Body Mass Index')
    DPF = st.number_input('Your Diabetes Pedigree Function')
      
    
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
    if prediction == 1:
        st.write(name," you either have diabetes or are likely to have it. Please visit the doctor as soon as possible.")
    else:
        st.write('Hurray!', name, 'You are diabetes FREE.')

