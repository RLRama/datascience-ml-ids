import streamlit as st

st.set_page_config(
    page_title="Desarrolladores",
    page_icon="🤖",
)

st.balloons()

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/man-technologist_1f468-200d-1f4bb.png",
    width=100
)
st.title('Equipo de trabajo')

tab1, tab2, tab3, tab4 = st.tabs(["Cano Ángel Rodrigo", "Dominguez Sotomayor Santiago Ismael", "Molina Leguiza Juan Cruz", "Rios Lopez Ramiro Ignacio"])

with tab1: ### Pagina Rodrigo
   st.image("CAR.png", width=200)
   st.markdown(
      """
      ### Nombre
      - Cano Ángel Rodrigo
      ---
      ### Matrícula
      - EISI821
      ---
      ### Fecha de Nacimiento
      - 07/11/00
      ---
      ### Redes Sociales
      - [Perfil en Instagram](https://www.instagram.com/cano.rodrigo10/)
      - [Perfil en Github](https://github.com/Canoro10)
      - [Perfil en Linkedin](https://www.linkedin.com/in/rodrigo-cano-4bb070219/)
      ---
      ### Correo Electrónico
      - rodrigocano765@gmail.com
      """
   )

with tab2: ### Pagina Santiago
   
   st.image("DSSI.png", width=200)
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

with tab3: ### Pagina Juan Cruz
   
   st.image("MLJC.png", width=200)
   st.markdown(
      """
      ### Nombre
      - Molina Leguiza Juan Cruz
      ---
      ### Matrícula
      - EISI767
      ---
      ### Fecha de Nacimiento
      - 29/03/01
      ---
      ### Redes Sociales
      - [Perfil en Instagram](https://www.instagram.com/juancruzmolina__/)
      - [Perfil en Github](https://github.com/JuanCruzMolina)
      - [Perfil en Linkedin](https://www.linkedin.com/in/juan-cruz-molina-6a512822a)
      ---
      ### Correo Electrónico
      - juancruzmolina01@gmail.com
      """
   )

with tab4: ### Pagina Ramiro
   
   st.image("RLRI.png", width=200)
   st.markdown(
      """
      ### Nombre
      - Rios Lopez Ramiro Ignacio
      ---
      ### Matrícula
      - EISI801
      ---
      ### Fecha de Nacimiento
      - 11/06/01
      ---
      ### Redes Sociales
      - [Perfil en Instagram](https://www.instagram.com/rama.riosl/)
      - [Perfil en Github](https://github.com/RLRama)
      - [Perfil en Linkedin](https://www.linkedin.com/in/ramiro-ignacio-rios-lopez-bb1006225/)
      ---
      ### Correo Electrónico
      - rl.ramiro11@gmail.com
      """
   )

st.markdown(
   """
   ### Información académica
    - Carrera: Ingeniería en Sistemas de Información
    - Cátedra: Seminario de Actualización II
    - Año académico: 2022
    - Prof.: Mg. Ms. Lic. Marcelo Fabio Roldán
    - Prof.: Lic. Cristina Gramajo
    - Prof.: Lic. Pablo Córdoba
   """
)