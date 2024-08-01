import streamlit as st

st.title('🎈 App Name')

user_input = st.text_area('Write something to activate the AI:', height=200)


if st.button('Generate Story'):
    with st.spinner('Generating Story...'):
        response = graph.stream({"question": user_input}, config, stream_mode="values") #model(user_input, max_length=max_length, num_return_sequences=num_sequences)
        for event in events:
          _print_event(event, _printed)
