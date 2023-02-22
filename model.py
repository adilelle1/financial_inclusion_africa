import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import io 

#1. Import
df = pd.read_csv('Train.csv')

#2. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Main menu',
        options=['Home', 'Plots', 'Model'],
    )

# Pagina 1 = Home
if selected == 'Home':
    st.title('Bienvenidos a Financial inclusion!')
    st.write('Vamos a trabajar con un dataset de inclusion financiera en africa')
    st.write("A continuacion podemos ver como esta compuesto de set de datos")
    st.write("Aqui tenemos un ejemplo de la informacion cruda disponible")
    st.dataframe(df.head())
    st.write("Como vemos hay varias columnas, algunas con datos numericos, otros categoricos y algunos redundantes.")
    st.subheader("\n Descripcion de columnas")
    st.markdown("\n **country** :  En qué ciudad vive")
    st.markdown("\n **bank_account** :  Si posee cuenta de banco o no -- objetivo")
    st.markdown("\n **location_type** :  Si vive en zona rural o urbana")
    st.markdown("\n **cellphone_access** :  Si tiene acceso a teléfono móvil")
    st.markdown("\n **household_size** :  Cantidad de personas que viven en el hogar")
    st.markdown("\n **age_of_respondent** :  Edad de la persona entrevistada")
    st.markdown("\n **gender_of_respondent** :  Género de la persona (Male, Female)")
    st.markdown("\n **relationship_with_head** :  Relación con el responsable de la casa")
    st.markdown("\n **education_level** :  Nivel de educación")
    st.markdown("\n **job_type** :  Tipo de trabajo del entrevistado")


# Pagina 2 = Graficos
elif selected == 'Plots':
    st.title('Plots')

    #countplot
    col_countplot = st.sidebar.selectbox('Countplot column',['country','location_type','household_size','relationship_with_head','marital_status','education_level','job_type'])
    def count_plot(): 
        fig = plt.figure(figsize=(15, 8))
        g = sns.countplot(data=df, y=col_countplot,palette="rainbow", order = df[col_countplot].value_counts().index).set_title(col_countplot,
                          fontdict = {'fontsize': 40,      
                                      'fontweight': 'bold', 
                                      'color': 'black'})
        st.pyplot(fig)


    # pie chart
    col_piechart = st.sidebar.selectbox('Pie chart column',['bank_account','gender_of_respondent','cellphone_access'])

    def pie_plot():
        fig = plt.figure(figsize=(10, 4))
        colors = sns.color_palette('rainbow')[0:5]
        data_pie = df[col_piechart].value_counts()
        if col_piechart == 'bank_account':
            labels = ['No tiene cuenta bancaria', 'Tiene cuenta bancaria']
        elif col_piechart == 'cellphone_access':
            labels = ['No tiene acceso a teléfono celular', 'Tiene acceso a teléfono celular']
        elif col_piechart == 'gender_of_respondent':
            labels = ['Hombre', 'Mujer']
        plt.pie(data_pie, labels = labels, colors = colors, autopct='%.0f%%')
        st.pyplot(fig)


    # Heatmap
    def heatmap_plot(): 
        fig = plt.figure(figsize=(15, 8))
        g = sns.heatmap(df.corr(),annot=True, cmap="YlGnBu")
        st.pyplot(fig)
    

    if __name__ == '__main__':
        st.header('Countplot')
        count_plot()
        st.header('Pie chart')
        pie_plot()
        st.header('Heatmap')
        heatmap_plot()

# Pagina 3 = Modelo
elif selected == 'Model':
    st.title('Ready to try your model?')
    def inputs():
        st.sidebar.header('Model inputs')
        country = st.sidebar.selectbox('Country', df.country.unique())	
        location_type = st.sidebar.selectbox('Location type', df.location_type.unique())
        cellphone_access = st.sidebar.selectbox('Cellphone access', df.cellphone_access.unique())
        household_size = st.sidebar.number_input('Household size', 1) 
        age_of_respondent = st.sidebar.number_input('Age', 16) 
        gender_of_respondent = st.sidebar.selectbox('Gender', df.gender_of_respondent.unique())
        relationship_with_head = st.sidebar.selectbox('Relationship with head', df.relationship_with_head.unique())
        marital_status = st.sidebar.selectbox('Marital status', df.marital_status.unique()) 
        education_level = st.sidebar.selectbox('Education level', df.education_level.unique())
        job_type = st.sidebar.selectbox('Job type', df.job_type.unique()) 
        button = st.sidebar.button('Try model!')
        return country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type, button

    def get_data(country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type):
            data_inputs = {'country': country, 
                    'location_type': location_type, 
                    'cellphone_access': cellphone_access, 
                    'household_size': household_size, 
                    'age_of_respondent':age_of_respondent, 
                    'gender_of_respondent': gender_of_respondent, 
                    'relationship_with_head': relationship_with_head, 
                    'marital_status': marital_status, 
                    'education_level': education_level, 
                    'job_type': job_type}
            data= pd.DataFrame(data_inputs, index=[0])
            return data



    def print_results():
        country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type, button = inputs()
        if button:
            st.header('Probando el modelo con los datos ingresados')
            st.write('') 
            trial_data = get_data(country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type)
            
            # imprimimos el df con los inputs
            st.write('Estos son los datos que ingresaste:')
            st.dataframe(trial_data)

            # LLamo al pkl con el modelo
            with open('./financial_inclusion.pkl', 'rb') as clf_inclusion:
                modelo_inclusion = pickle.load(clf_inclusion)
            # Prediccion usando el trial data con lo insertado en el form
            if modelo_inclusion.predict(trial_data) == 1:
                st.write('---')
                st.markdown('<h4 style="text-align: center">El individuo ya se encuentra bancarizado</h4>',unsafe_allow_html=True)
                st.write('---')
            else:
                st.write('---')
                st.markdown('<h4 style="text-align: center">El individuo no se encuentra bancarizado</h4>', unsafe_allow_html=True)
                st.write('---')

                


    if __name__ == '__main__':
        print_results()
