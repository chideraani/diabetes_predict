# Diabetes Prediction
Predict whether or not patients have diabetes using machine learning. Deploy model using Streamlit.

## Dataset 
Pima Indians Diabetes dataset was obtained from [Kaggle](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906/data). The target attribute is the column labeled 'Outcome'. Columns in the dataset are explained below:
- **Pregnancies:** The number of times the person was pregnant
- **Glucose:** This is the concentration of glucose present in the blood (plasma). For a healthy person (a non-diabetic), the normal glucose concentration is between 4.0 to 5.4 mmol/L (72 to 99 mg/dL) and up to 7.8 mmol/L (140 mg/dL) 2 hours after eating.
- **Blood Pressure:** Diastolic blood pressure (mm Hg). A normal blood pressure should be less than or equal to 120/80 mmHg.
- **SkinThickness:** Triceps skin fold thickness (mm). The triceps skinfold is necessary for calculating the upper arm muscle circumference. Its thickness gives information about the fat reserves of the body. For adults, the standard normal values for triceps skinfolds are 2.5mm (men) or about 20% fat; 18.0mm (women) or about 30% fat.
- **Insulin:** 2-Hour serum insulin (mu U/ml). The normal insulin level range 2 hours after glucose administration is 16-166 mIU/L.
- **BMI:** Body mass index (weight in kg/ (height in m²)).BMI is a measure of body fat based on height and weight that applies to adult men and women. An underweight person has a BMI of 18.5 or less, a normal weight person has a BMI range of 18.5-24.9, an overweight person has a BMI in the range of 25-29.9, and an obese person has a BMI of 30 or greater.
- **DiabetesPedigreeFunction:** A diabetes pedigree function is a function which scores likelihood of diabetes based on family history.
- **Outcome (target variable):** 0-no diabetes, 1-has diabetes.

## Requirements
- Install requirements using `pip install -r requirements.txt`
  - Ensure you use Python 3.

## Output
- The best model proved to the Gradient Boosting Classifier with 89.6% accuracy.
- A website application was built from this project using Streamlit and deployed to Heroku. View [here](https://diadetect.herokuapp.com/)

## Extending this...
- Try to carry out an extensive EDA and maybe create more features.
- Try to increase the model accuracy by hyperparameter tuning or other methods.
