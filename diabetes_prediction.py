#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 


# In[41]:


# ML Libraries
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


# In[3]:


# Setting figure size
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size


# In[4]:


# Get the data
df = pd.read_csv("diabetes.csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


# Count of the outcome variable
g = sns.countplot(x='Outcome',data=df, palette='pastel')
plt.title('Count of Outcome Variable')
plt.xlabel('Outcome')
plt.ylabel('Count')

for p in g.patches:
    g.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',                    va = 'center', xytext = (0, 10), textcoords = 'offset points')


# In[8]:


# replacting 0 with nan
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.head(3)


# In[9]:


# Distribution of all variables
f, ax= plt.subplots(figsize=(15, 10))

ax.set(xlim=(-.05, 200))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data = df, orient = 'v', palette = 'Set2')


# In[10]:


# Correlation plot
f, ax = plt.subplots(figsize=(11, 9))

mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.heatmap(df.corr(), mask=mask, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[11]:


# Checking for null values
df.isnull().sum()


# ## Replacing missing values

# In[12]:


# function to find the mean 
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
    return temp


# #### Glucose

# In[13]:


median_target("Glucose")


# In[14]:


df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3


# #### Blood pressure

# In[15]:


median_target("BloodPressure")


# In[16]:


df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3


# #### Skin thickness

# In[17]:


median_target("SkinThickness")


# In[18]:


df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0


# #### Insulin

# In[19]:


median_target("Insulin")


# In[20]:


df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8


# #### BMI

# In[21]:


median_target("BMI")


# In[22]:


df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4


# ## Exploratory Data Analysis

# In[23]:


# Distribution plot for each column
for i in df.columns:
    sns.distplot(df[i], hist=True, kde=True)
    plt.show()


# In[24]:


# Histogram for each column
plt.rcParams["figure.figsize"] = (20, 10)
df.hist(grid=False, alpha=0.5)


# In[25]:


# Glucose vs BP 
plt.rcParams["figure.figsize"] = (10, 8)
sns.scatterplot(x='Glucose', y='BloodPressure', hue='Outcome', data=df, s=60, alpha=0.8)
plt.title('Glucose vs Blood Pressure')


# In[26]:


# Insulin vs Blood Pressure 
plt.rcParams["figure.figsize"] = (10, 8)
sns.scatterplot(x='Insulin', y='BloodPressure', hue='Outcome', data=df, s=60, alpha=0.8)
plt.xticks([0, 166, 200, 400, 600])
plt.title('Insulin vs Blood Pressure')


# In[27]:


# Glucose vs Age
plt.rcParams["figure.figsize"] = (10, 8)
sns.scatterplot(x='Glucose', y='Age', hue='Outcome', data=df, s=60, alpha=0.8)
plt.title('Glucose vs Age')


# In[28]:


# BMI vs Age
plt.rcParams["figure.figsize"] = (10, 8)
sns.scatterplot(x='BMI', y='Age', hue='Outcome', data=df, s=60, alpha=0.8)
plt.xticks([0,15, 20, 25, 30, 40, 50, 60])
plt.title('BMI vs Age')


# In[29]:


# Skin Thickness vs DPF
plt.rcParams["figure.figsize"] = (10, 8)
sns.scatterplot(x='SkinThickness', y='DiabetesPedigreeFunction', hue='Outcome', data=df, s=60, alpha=0.8)
plt.title('Skin Thickness vs DPF')


# # Model Building

# In[30]:


# splitting columns
X = df.drop(columns='Outcome')
y = df['Outcome']


# In[31]:


#scaling
scaler = StandardScaler()
X =  pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[32]:


# Split the dataset into 70% Training set and 30% Testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[54]:


# Function for accessing performance:
def classification_model(y_test, prediction, model, x_train, y_train):
    # accuracy score
    accuracy = metrics.accuracy_score(y_test,prediction)
    print ("Accuracy Score: %s" % "{0:.3%}".format(accuracy))
    # F1 score
    f1 = metrics.f1_score(y_test,prediction)
    print ("F1 Score: %s" % "{0:.3%}".format(f1))



    #cross validation with 5 folds
    kf = KFold(n_splits=5)
    kf.split(x_train)    

    accuracy_model = []
    
    for train_index, test_index in kf.split(x_train):
            # Split train-test
            X_train, X_test = x_train.iloc[train_index], x_train.iloc[test_index]
            Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]
            # Train the model
            model.fit(X_train, Y_train)
            # Append to accuracy_model the accuracy of the model
            accuracy_model.append(accuracy_score(Y_test, model.predict(X_test)))
    
    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(accuracy_model)))
    
    
# Function for confusion matrix plot
def confusion_matrix_plot (y_test, prediction):
    
    cm = confusion_matrix(y_test, prediction)
    classes = ['0', '1']
    figure, ax = plot_confusion_matrix(conf_mat = cm,
                                       class_names = classes,
                                       show_absolute = True,
                                       show_normed = False,
                                       colorbar = True)

    plt.show()


# ## Logistic Regression Model

# In[55]:


logmodel = LogisticRegression()

logmodel.fit(x_train, y_train)

prediction = logmodel.predict(x_test)

classification_model(y_test, prediction, logmodel, x_train, y_train)
confusion_matrix_plot(y_test, prediction)


# ## Decision Tree Model

# In[56]:


dec_tree = DecisionTreeClassifier()

dec_tree.fit(x_train, y_train)

prediction2 = dec_tree.predict(x_test)

classification_model(y_test, prediction2, dec_tree, x_train, y_train)
confusion_matrix_plot(y_test, prediction2)


# ## Random Forest Model

# In[57]:


rfmodel = RandomForestClassifier(random_state=1)

rfmodel.fit(x_train, y_train)

prediction3 = rfmodel.predict(x_test)

classification_model(y_test, prediction3, rfmodel, x_train, y_train)
confusion_matrix_plot(y_test, prediction3)


# ## Gradient Boosting Model

# In[58]:


gb = GradientBoostingClassifier()

gb.fit(x_train, y_train)

prediction4 = gb.predict(x_test)

classification_model(y_test, prediction4, gb, x_train, y_train)
confusion_matrix_plot(y_test, prediction4)


# ## Feature Importance

# In[62]:


feature_importances = gb.feature_importances_ 

plt.barh(x_train.columns, feature_importances)
plt.xlabel('Feature Importances')
plt.ylabel('Feature Labels')
plt.title('Feature Importance of Variables')
plt.show()


# In[ ]:




