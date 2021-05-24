import streamlit as st
from multiapp import MultiApp
from apps import linear, integer# import your app modules here

st.set_page_config(layout="wide")
app = MultiApp()


# Add all your application here
app.add_app("Linear model", linear.app)
app.add_app("Integer model", integer.app)

# The main app
app.run()
