import streamlit as st
import os
from codeforllama import test_poop

st.title('Yodl')




user_input = st.text_area('Write something to activate the AI:', height=200)


if st.button('Generate Story'):
    with st.spinner('Generating Story...'):
        response = test_poop(user_input)
