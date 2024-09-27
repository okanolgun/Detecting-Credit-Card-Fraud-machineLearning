import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#loading data 
data = pd.read_csv('creditcard.csv')

#separate 'class' columns
legit = data['Class'==0]
fraud = data['Class'==1]

#balancing the numbers of diff dfs
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample,fraud], axis=0)

#splitting data
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

#training 
model = LogisticRegression()
model.fit(X_train, y_train)

#evalıate model score 
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

#streamlitapp 
st.title("Creadit Card Fraud Detection")
input_df = st.text_input("Enter the following features")
input_df_splitted = input_df.split(',')

submit = st.button("submit")

if submit : 
    features = np_df = np.asarray(input_df_splitted, dtype=np.float64)
    