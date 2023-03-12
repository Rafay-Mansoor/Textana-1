import streamlit as st

st.title('Text Analytics Assignment - Rafay')

title = st.text_input('Enter Query', 'Karachi witnessed decrease in terrorism')
st.write('The current query is:', title)
