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

import pandas

st.write(pandas)
