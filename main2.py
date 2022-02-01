import sys
import re
# from typing import final
# import utils_for_dashboard
import pickle
import numpy as np
import pandas as pd
from io import StringIO
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score
# import matplotlib.pyplot as plt
import streamlit as st
# import plotly.express as px
# import plotly.subplots as sp
# import plotly.figure_factory as ff
# import plotly.graph_objects as go

# from collections import Counter
# from nltk import ngrams
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

from io import BytesIO
import requests

st.set_page_config(layout = "wide")

# Import data
# train=pd.read_csv('https://raw.githubusercontent.com/nurchamid/dashboard_streamlit/main/data-registry/train.csv')
# train['dataset'] = 'train'
# train = utils_for_dashboard.load_data()

# Import pickle model
# mLink = 'https://github.com/teamAA/nlpproject2021_dep/blob/main/src/model_v1.pkl'
# mfile = BytesIO(requests.get(mLink).content)
# loaded_model = pickle.load(mfile)

# Import data for prediction
# pred_train = train[['text','sentiment', 'dataset']]

# Preprocess
# data=utils_for_dashboard.data_preproc_v1(train)
start_date=pd.to_datetime('2021-08-01')
end_date=pd.to_datetime('2021-08-08')
# data["tweet_date"] = np.random.choice(pd.date_range(start_date, end_date), len(data))
# list_sentiment=['all']+list(data['sentiment'].unique())

# Create title for all pages
st.title('Sentiment Analysis')
st.markdown('The project is to build a model that will determine the tone (neutral, positive, negative) of the tweet text.')