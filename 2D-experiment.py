import streamlit as st
from multiapp import MultiApp
from apps import linear, integer, integer_salvo, test, test2, decomposed_choquet_L1, decomposed_choquet_Linf # import your app modules here

st.set_page_config(layout="wide")
app = MultiApp()


# Add all your application here
#app.add_app("Linear model", linear.app)
#app.add_app("Integer model (Yannik)", integer.app)
#app.add_app("Integer model (Salvo)", integer_salvo.app)
app.add_app("Choquet min-norm", decomposed_choquet_L1.app)
app.add_app("Choquet Tschebycheff (max-)norm", decomposed_choquet_Linf.app)

# The main app
app.run()
