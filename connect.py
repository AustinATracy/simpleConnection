import streamlit as st

from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import udf, col
from snowflake.snowpark.types import Variant
from snowflake.snowpark import functions as fn

from snowflake.snowpark import version

#horray

st.write(version.VERSION)

credentials = {
  "account": "AITNLHM-TYB61838",
  "user": "AustinTracy7",
  "password": st.secrets["password"],
  "role": "ACCOUNTADMIN",
  "database": "IMDB",
  "schema": "PUBLIC",
  "warehouse": "COMPUTE_WH"
}

session = Session.builder.configs(credentials).create()

session.add_packages("scikit-learn", "pandas", "numpy", "nltk", "joblib", "cachetools")

st.session_state['snowpark_session'] = session

# import pandas

# st.write(pandas)

# from nltk.corpus import stopwords
st.write(nltk)
# import sklearn.feature_extraction.text as txt
# from sklearn import svm
# import os
# from joblib import dump
    
# train_dataset = session.table("IMDB.PUBLIC.TRAIN_DATASET")
# train_dataset_flag = train_dataset.withColumn("SENTIMENT_FLAG", fn.when(train_dataset.SENTIMENT == "positive", 1)
#                                   .otherwise(2))
# train_x = train_dataset_flag.toPandas().REVIEW.values
# train_y = train_dataset_flag.toPandas().SENTIMENT_FLAG.values
# st.write('Taille train x : ', len(train_x))
# st.write('Taille train y : ', len(train_y))

