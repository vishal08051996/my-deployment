# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:07:37 2022

@author: LENOVO
"""





import streamlit as st
import pickle
import numpy as np
import pandas as pd

#load the model and dataframe
df = pd.read_csv("G:/data science/360PROJECT_2/venky_duplicate/data_p2.csv",encoding='latin1')
pipe = pickle.load(open("G:/data science/360PROJECT_2/venky_duplicate/data_model_p2.pkl", "rb"))

st.title("Cost prediction of transport")

#Now we will take user input one by one as per our dataframe


engine_type = st.selectbox('engine_type', df['engine_type'].unique())
Train_Type = st.selectbox('Train_Type', df['Train_Type'].unique())


Train_bogies = st.sidebar.number_input('Train_bogies')

speed_of_train = st.selectbox("speed_of_train", df['speed_of_train'].unique())


source_station = st.selectbox("source_station", df['source_station'].unique())
destination_station = st.selectbox("destination_station", df['source_station'].unique())


carry_weight = st.sidebar.number_input('carry_weight')
maintance_cost = st.sidebar.number_input('maintance_cost')
Hours = st.sidebar.number_input('Hours')


#Prediction
if st.button('Price'):
    query = np.array([Institute,Subject,Location, Trainer_Qualification, Trainer_experiance, Online_classes, Offline_classes, Course_hours,
                      Course_level,Course_rating,Rental_permises, Trainer_slary,Maintaince_cost,Non_teaching_staff_salary, Placements])
    query = query.reshape(1, 15)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of the course" + prediction)
    