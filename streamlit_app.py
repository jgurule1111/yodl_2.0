import streamlit as st
import os
import codeforllama

st.title('Yodl')

def test_poop(questions):
    config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
     }
    }
    question = str(questions)
    event = graph.invoke({"question": questions}, config)

    message = event.get("messages")

    if message:
        if isinstance(message, list):
            message = message[-1]
            msg_repr = message.pretty_repr(html=True)

    return print(msg_repr)

    



    
#user_input = st.text_area('Write something to activate the AI:', height=200)
user_input = st.text_input('Enter your question:', placeholder = 'ex. what are the financial highlights for the year?')


if st.button('Enter'):
    with st.spinner('Thinking...'):
        response = test_poop(user_input)
        st.write(response)
