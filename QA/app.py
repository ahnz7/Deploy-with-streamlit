#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Time    :   2021/01/24 10:06:15
@Author  :   Hanlin Li 
@Version :   1.0
@Contact :   ahnz830@gmail.com
'''

# here put the import lib

import streamlit as st
from transformers import *
import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys 
from transformers import *
import torch
import pandas as pd
from QA_model import QA
import os
import seaborn as sns

#! sidebar: model choose, file-path, question, first n. answers, 

#select Model
st.title('QA-System')
model_name = st.sidebar.selectbox('Model Select',('QA','Stichwort+QA'))
st.header("""
Model
        """)
st.write(model_name)

if model_name == 'QA':
        model = QA()

if model_name == 'Stichwort+QA':
    metric = st.sidebar.selectbox('Metric Similarity',('Euclidean Distance','Cosine similarity'))
    st.subheader("""
Metric Similarity
        """)
    st.write(metric)

# file path
filename = st.sidebar.text_input('Enter a file path:')
# c_button = st.sidebar.button('submit')


#question
question = st.sidebar.text_input('Question')
q_button = st.sidebar.button('submit')
st.header("""
Question : 
        """)
st.write(question)
        
if question and q_button:
        f = open (filename)
        dataset = f.read()
        st.header('Context: ')
        st.write(dataset)
        st.sidebar.success('Success')

#n. answers
n = st.sidebar.slider('No. of answers', min_value=1, max_value=10)
st.header("""
No. of Answers
        """)
st.write(n)
if q_button:
        answers = model.predict(questions = question,context = context,n = n)
        df = pd.DataFrame(columns = ['score','start','end','answer'])
        if n == 1:
                df = df.append(answers,ignore_index = True)
                st.dataframe(df)
        else:
                for answer in answers:
                        df = df.append(answer,ignore_index=True)
                st.table(df)
                df_index = df[['start','end']]
                st.title('Scatter')
                fig, ax = plt.subplots()
                ax = sns.swarmplot(x = df_index['start'],y = df_index['end'],hue = df.index,data = df_index)
                st.pyplot(fig)
                
                st.title('Heatmap')
                df_index['start'] = pd.Series(df['start']).astype(int)
                df_index['end'] = pd.Series(df['end']).astype(int)
                fig, ax = plt.subplots()
                ax = sns.heatmap(df_index,annot=True)
                st.pyplot(fig)
                
