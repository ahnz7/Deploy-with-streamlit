#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   QA_model.py
@Time    :   2021/01/24 10:06:43
@Author  :   Hanlin Li 
@Version :   1.0
@Contact :   ahnz830@gmail.com
'''

# here put the import lib

from transformers import *
import os

class QA():
    def __init__(self):
        self.pipeline = pipeline('question-answering',model = 'ktrapeznikov/albert-xlarge-v2-squad-v2')
    
    def predict(self,questions,context,n):
        answers = self.pipeline(context = context,question = questions,topk = n,max_answer_len = 50,doc_stride = 256)
        return answers