import streamlit as st
from src.utils.prediction import load_and_cache_resources

st.set_page_config("Duplicate checker", page_icon=':question:')
st.title('Questions Duplicate Checker')
st.divider()
question1 = st.text_input('Enter Question 1:')
question2 = st.text_input('Enter Question 2:')
submit = st.button('Check Duplicate', type='primary')

# Handling inputs
if submit and question1 != '' and question2 != '':
    model_handler = load_and_cache_resources()
    prediction_result = model_handler.predict_on_questions(question1, question2)
    st.write(':blue[Result:]',prediction_result)