import streamlit as st
import pandas as pd
import dvc.api
st.set_page_config(layout = "wide")

st.title('Sentiment Analysis')
st.markdown('The project is to build a model that will determine the tone (neutral, positive, negative) of the tweet text.')

data = pd.read_csv(dvc.api.get_url('data_model/train.csv'))
total=data.shape[0]
st.write('Total tweets : ',total)

data2 = data.head()
st.table(data2)