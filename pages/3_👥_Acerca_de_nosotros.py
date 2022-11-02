import streamlit as st

st.set_page_config(
    page_title="Desarrolladores",
    page_icon="🤖",
)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/man-technologist_1f468-200d-1f4bb.png",
    width=100
)
st.title('Equipo de trabajo')

tab1, tab2, tab3, tab4 = st.tabs(["Cano Ángel Rodrigo", "Dominguez Sotomayor Santiago Ismael", "Molina Leguiza Juan Cruz", "Rios Lopez Ramiro Ignacio"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab4:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)