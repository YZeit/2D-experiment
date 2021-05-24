import streamlit as st
from multiapp import MultiApp
from apps import linear, integer# import your app modules here

st.set_page_config(layout="wide")
app = MultiApp()


# Add all your application here
app.add_app("Mixed-Integer Linear programming model (MILP)", linear.app)
app.add_app("Integer Linear programming model (ILP)", integer.app)

# The main app
app.run()
