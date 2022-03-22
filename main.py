import streamlit as st 
import numpy as np 
import pandas as pd 
import streamlit.components.v1 as components

# ML Libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# heading
st.markdown("<h1 style='text-align: center; color: blue;'>DIADETECT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>...a diabetes detection system</h4><br>", unsafe_allow_html=True)


st.write("Diabetes is a chronic disease that occurs when your blood glucose is too high. This application helps to effectively detect if someone has diabetes using Machine Learning and Python. " )



#Get the data
df = pd.read_csv("diabetes.csv")

# replacting 0 with nan
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# replacing missing values

# function to find the mean 
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
    return temp


# Glucose
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3

# Blood pressure
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3

# Skin thickness
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0

# Insulin
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8

# BMI
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4



# splitting columns
X = df.drop(columns='Outcome')
y = df['Outcome']


#scaling
scaler = StandardScaler()
X =  pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])


# Split the dataset into 70% Training set and 30% Testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

name = st.text_input('What is your name?').capitalize()

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
    gb = GradientBoostingClassifier(random_state=1)
    gb.fit(x_train, y_train)
    prediction = gb.predict(user_input)
    

    if prediction == 1:
        st.write(name,", you either have diabetes or are likely to have it. Please visit the doctor as soon as possible.")
        
    else:
        st.write('Hurray!', name, 'You are diabetes FREE.')

