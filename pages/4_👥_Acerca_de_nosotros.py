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
   
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
      st.markdown(
      """
      ### Nombre
      - Cano Angel Rodrigo
      ---
      ### Matrícula
      - EISI821
      ---
      ### Fecha de Nacimiento
      - 07/11/2000
      ---
      ### Redes Sociales
      - [Perfil en Instagram](https://www.instagram.com/cano.rodrigo10/)
      - [Perfil en Github](https://www.linkedin.com/in/rodrigo-cano-4bb070219/)
      - [Perfil en Linkedin](https://github.com/Canoro10)
      ---
      ### Correo Electrónico
      -  rodrigocano765@gmail.com
      """
   )

with tab2:
   
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
   st.markdown(
      """
      ### Nombre
      - Dominguez Sotomayor Santiago Ismael
      ---
      ### Matrícula
      - EISI782
      ---
      ### Fecha de Nacimiento
      - 04/05/01
      ---
      ### Redes Sociales
      - [Perfil en Instagram](https://www.instagram.com/santidominguez01/)
      - [Perfil en Github](https://github.com/SantiDominguez1)
      - [Perfil en Linkedin](https://www.linkedin.com/in/santiago-ismael-dominguez-sotomayor-a55009225//)
      ---
      ### Correo Electrónico
      - santiuni04@gmail.com
      """
   )

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab4:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

