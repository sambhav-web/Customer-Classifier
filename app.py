import streamlit as st
import pandas as pd
import numpy as np
import pickle
with open('le.pkl','rb') as f:
  le=pickle.load(f)
with open('scaler.pkl','rb') as f:
  scaler=pickle.load(f)
with open('pca.pkl','rb') as f:
  pca=pickle.load(f)
with open('kmean.pkl','rb') as f:
  kmean=pickle.load(f)
df=pd.read_csv('mall.csv')
st.title('Mall Customer Segmentation')
sex=st.selectbox('Select Gender',list(df['Genre'].unique()))
age=st.slider('Enter Age',df['Age'].min(),df['Age'].max())
income=st.number_input('Enter Annual Income',df['Annual Income (k$)'].min(),df['Annual Income (k$)'].max())
score=st.number_input('Enter Spending Score',df['Spending Score (1-100)'].min(),df['Spending Score (1-100)'].max())
def categorize(sex,age,income,score):
  input={
      'Genre':[sex],
      'Age':[age],
      'Annual Income (k$)':[income],
      'Spending Score (1-100)':[score]
  }
  input=pd.DataFrame(input)
  input['Genre']=le.transform(input['Genre'])
  input[['Age','Annual Income (k$)','Spending Score (1-100)']]=scaler.transform(input[['Age','Annual Income (k$)','Spending Score (1-100)']])
  input=pca.transform(input)
  input=kmean.predict(input)[0]
  if input == 0:
    return ('High Earning, Low Spending')
  elif input == 1:
    return ('Low Earning, High Spending')
  elif input == 2:
    return ('High Earning, High Spending')
  else:
    return ('Low Earning, Low Spending')
if st.button('User Class'):
  category=categorize(sex,age,income,score)
  st.success(f"User belongs to {category}")