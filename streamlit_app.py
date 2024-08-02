import streamlit as st
import os
import pickle

st.title('🎈 Yodl')
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

model = pickle.load(open(f'{working_dir}/code/code1', 'rb'))

user_input = st.text_area('Write something to activate the AI:', height=200)


if st.button('Generate Story'):
    with st.spinner('Generating Story...'):
        response = graph.stream({"question": user_input}, config, stream_mode="values") #model(user_input, max_length=max_length, num_return_sequences=num_sequences)
        for event in events:
          _print_event(event, _printed)
