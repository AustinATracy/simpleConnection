import streamlit as st

from snowflake.snowpark import version
from snowflake.snowpark.session import Session

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

st.write(session)
