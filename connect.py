import streamlit as st
from streamlit_option_menu import option_menu

from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import udf, col
from snowflake.snowpark.types import Variant
from snowflake.snowpark import functions as fn

import pandas as pd
import numpy as np

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

st.write(session)
