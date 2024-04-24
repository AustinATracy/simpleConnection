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

session

st.session_state['snowpark_session'] = session

# import pandas

# st.write(pandas)
# import nltk
# from nltk.corpus import stopwords
# st.write(stopwords)
import sklearn.feature_extraction.text as txt
from sklearn import svm
# import os
from joblib import dump
    
# train_dataset = session.table("IMDB.PUBLIC.TRAIN_DATASET")
# st.write(train_dataset)
# train_dataset_flag = train_dataset.withColumn("SENTIMENT_FLAG", fn.when(train_dataset.SENTIMENT == "positive", 1)
#                                   .otherwise(2))
# train_x = train_dataset_flag.toPandas().REVIEW.values
# train_y = train_dataset_flag.toPandas().SENTIMENT_FLAG.values
# st.write('Taille train x : ', len(train_x))
# st.write('Taille train y : ', len(train_y))








from snowflake.snowpark.context import get_active_session

# Get the current credentials
session = get_active_session()

from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import udf, col
from snowflake.snowpark.types import Variant
from snowflake.snowpark import functions as fn
from snowflake.snowpark import version
session.add_packages("scikit-learn", "pandas", "numpy")

import numpy as np
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
import sklearn.feature_extraction.text as txt
from sklearn import svm
        
test_x = np.array(["that was incredible. I'm speechless","it was the worst ever","it was okay. I might go see it again","What was this garbage. I don't care to know."]) 

train_dataset = session.sql("SELECT * FROM IMDB_ACCESIBLE.PUBLIC.TRAIN_DATASET").collect()
train_dataset_df = pd.DataFrame(train_dataset)
train_dataset_df["SENTIMENT_FLAG"] = 2
train_dataset_df.loc[train_dataset_df["SENTIMENT"] == "POSITIVE","SENTIMENT_FLAG"] = 1
# train_dataset_flag = train_dataset.withColumn("SENTIMENT_FLAG", fn.when(train_dataset.SENTIMENT == "positive", 1).otherwise(2))

train_x = train_dataset_df.fillna(" ").REVIEW.values
train_y = train_dataset_df.SENTIMENT_FLAG.values

st.write('Taille train x : ', len(train_x))
st.write('Taille train y : ', len(train_y))
    
st.write('Configuring parameters ...')
# bags of words: parametrage
analyzer = u'word' # {‘word’, ‘char’, ‘char_wb’}
ngram_range = (1,2) # unigrammes
token = u"[\\w']+\\w\\b" #
max_df=0.02    #50. * 1./len(train_x)  #default
min_df=1 * 1./len(train_x) # on enleve les mots qui apparaissent moins de 1 fois
binary=True # presence coding
svm_max_iter = 100
svm_c = 1.8
    
st.write('Building Sparse Matrix ...')
vec = txt.CountVectorizer(
    token_pattern=token, \
    ngram_range=ngram_range, \
    analyzer=analyzer,\
    max_df=max_df, \
    min_df=min_df, \
    vocabulary=None, 
    binary=binary)

# pres => normalisation
# st.write(train_x)
bow = vec.fit_transform(train_x)

# st.write(type(vec.transform(test_x)))
st.write('Taille vocabulaire : ', len(vec.get_feature_names_out()))
    
st.write('Fitting model ...')
model = svm.LinearSVC(C=svm_c, max_iter=svm_max_iter)
model.fit(bow, train_y)
    
#### Create a stage to store the model
session.sql("CREATE STAGE IF NOT EXISTS MODELS").collect()
    
# Upload the Vectorizer (BOW) to a stage
st.write('Upload the Vectorizer (BOW) to a stage')
model_output_dire = '/tmp'
# model_file = os.path.join(model_output_dire, 'vect_review.joblib')
# dump(vec, model_file, compress=True)
# session.file.put("vect_review.joblib", "@MODELS", auto_compress=False, overwrite=True)
    
# Upload trained model to a stage
st.write('Upload trained model to a stage')
model_output_dire = '/tmp'
# model_file = os.path.join(model_output_dire, 'model_review.joblib')
# dump(model, model_file, compress=True)
# session.file.put("model_review.joblib", "@MODELS", auto_compress=False, overwrite=True)
    
st.write({"STATUS": "SUCCESS", "R2 Score Train": str(model.score(bow, train_y))})

st.write(test_x)
st.write(model.predict(vec.transform(test_x)))
