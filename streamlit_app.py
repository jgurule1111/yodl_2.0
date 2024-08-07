import streamlit as st
from codeforllama import test_poop

st.title('Yodl')



    
#user_input = st.text_area('Write something to activate the AI:', height=200)
user_input = st.text_input('Enter your question:', placeholder = 'ex. what are the financial highlights for the year?')


if st.button('Enter'):
    with st.spinner('Thinking...'):
        response = test_poop(user_input)
        st.write(response)
