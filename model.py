import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder


#1. Import
df = pd.read_csv('Train.csv')
df.drop(columns=['uniqueid', 'year'],inplace=True)

#2. Titulo de pagina
st.set_page_config(page_title="Financial inclusion prediction App")

#3. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Plots', 'Model backstage', 'Model trial'],
    )

# Pagina 1 = Home
if selected == 'Home':
    st.title('Financial inclusion in Africa')
    st.write('Predict who in Africa is most likely to have a bank account')


    st.header('Problemática')
    st.write('La inclusión financiera refiere al acceso que tienen las personas y las empresas a diversos productos y servicios financieros útiles y asequibles que atienden sus necesidades. Representa una preocupación global, ya que se considera como elemento facilitador para reducir la pobreza extrema y promover el crecimiento y desarrollo económico.')
    st.write('El acceso a cuentas bancarias, impacta tanto en el desarrollo humano, como en el ámbito económico, ya que permite a los hogares ahorrar, realizar pagos, acceder a créditos, financiamiento, entre otros; al mismo tiempo que ayuda a las empresas a aumentar su solvencia crediticia y mejorar su acceso a préstamos, seguros y servicios relacionados.')
    st.write('En África, la incusión financiera constituye uno de los principales problemas, ya que en una población compuesta de alrededor de 172,19 millones de personas (The World Bank Group , 2020) en 4 países (Kenia, Ruanda, Tanzania y Uganda), solo el 14% de la población adulta, representada por 9,1 millones, tiene este acceso (Zindi).')
    st.write('Por esta razón, la Nueva Alianza para el Desarrollo de África (NEPAD) ha involucrado a representantes del continente para encontrar soluciones que podrían mejorar la inclusión financiera y el bienestar de las personas que viven en África, utilizando el modelo empresarial cooperativo.')
    
    st.write('El conjunto de datos utilizado, ha sido extraído de Zindi, una red profesional para científicos de datos en África; constituyen los resultados de las encuestas de Finscope de 2016 a 2018.')
    st.write("[Zindi website](https://zindi.africa/competitions/financial-inclusion-in-africa)")

    st.write('Su contenido principal está relacionado con la información demográfica y servicios financieros utilizados por aproximadamente 23.500 personas en África, correspondientes a 4 países: Kenia, Ruanda, Tanzania y Uganda.')
    
    st.write("A continuacion podemos ver como esta compuesto el set de datos")
    st.dataframe(df.head())

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

    #histplot
    col_histplot = st.sidebar.selectbox('Histplot column',['country','location_type', 'age_of_respondent','household_size','relationship_with_head','marital_status','education_level','job_type'])
    def graf_hist():
        fig = px.histogram(df, x= col_histplot, color=col_histplot, color_discrete_sequence=px.colors.qualitative.Set2).update_xaxes(categoryorder='total descending')
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            bargap= 0.2,
            title={
                'text': (f'Distribución: {col_histplot}'),
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
        
            legend_title= None,
            font=dict(size=18)
            )
        st.plotly_chart(fig)


    # pie chart
    col_piechart = st.sidebar.selectbox('Pie chart column',['bank_account','gender_of_respondent','cellphone_access'])
    def graf_pie():

        fig = px.pie(df, names=col_piechart, color=col_piechart, color_discrete_sequence=px.colors.qualitative.Set2, hole=.5)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            title={
                'text': (f'Proporciones: {col_piechart}'),
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            legend_title= None,
            font=dict(size=18)
            )

        st.plotly_chart(fig)

    # boxplot
    col_box_plot = st.sidebar.selectbox('Boxplot column',['age_of_respondent','household_size'])
    def boxplot():
        fig = px.box(df, x=col_box_plot, color='bank_account')
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            title={
                'text': ('Boxplot: Edades'),
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(size=18)
            )
        st.plotly_chart(fig)


    # Heatmap
    def heatmap_plot():
        # Encodeamos las variables, facilitando un mejor análisis de las mismas
        binary_encoder = OneHotEncoder(sparse=False, drop='if_binary')
        data =df.copy()
        data['bank_account'], data['location_type'], data['cellphone_access'], data['gender_of_respondent'] = binary_encoder.fit_transform(data[['bank_account','location_type', 'cellphone_access','gender_of_respondent']]).T


        fig = px.imshow(data.corr(), text_auto=True, aspect="auto", color_continuous_scale='darkmint').update_xaxes(tickangle=45)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=800,
            bargap=0.2,
            title={
                'text': ('Heatmap: Variables numéricas'),
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}, 
            showlegend=False,
            font=dict(size=15)
            )


        st.plotly_chart(fig)

    

    if __name__ == '__main__':
        st.header('Histplot')
        graf_hist()
        st.header('Pie chart')
        graf_pie()
        st.header('Boxplot')
        boxplot()
        st.header('Heatmap')
        heatmap_plot()



# Pagina 3 = Comparación de modelos
elif selected == 'Model backstage':
    def model_backstage():
        st.title('Building a classification model')
        st.write('Para armar el modelo de clasificación, en primer lugar decidimos probar por separado como performaba cada clasificador y hacer una búsqueda de los parámetros óptimos de cada uno.')

        def print_model_comparison():
            comparacion_modelos_data = pd.read_csv('comparacion_modelos.csv', index_col='Unnamed: 0')
            comparacion_modelos_data_sorted = comparacion_modelos_data.sort_values(by='accuracy', ascending=False)
            return st.dataframe(comparacion_modelos_data_sorted)
        print_model_comparison()


        st.write('Vemos que todos los clasificadores tienen métricas similares.')
        st.write('Para nuestro modelo vamos a tomar dos, en este caso como los mejores fueron Gradient Boosting y XG Boost tomaremos esos.')

        st.write('Usando Pipeline y GridSearch obtuvimos los siguientes hiperparametros para nuestro modelo:')
        def print_model_params():
            moodel_params_data = pd.read_csv('model_params.csv', index_col='Unnamed: 0')
            return st.dataframe(moodel_params_data)
        print_model_params()

        st.write('Los resultados del modelo de clasificación fueron los siguientes:')
        def print_model_scores():
            moodel_scores_data = pd.read_csv('model_scores.csv', index_col='Unnamed: 0')
            return st.dataframe(moodel_scores_data)
        print_model_scores()

        st.write('Matriz de confusión:')
        st.image('heatmap.png')


        st.write('curva ROC-AUC:')
        st.image('roc_auc.png')

    if __name__ == '__main__':
        model_backstage()



# Pagina 4 = Modelo
elif selected == 'Model trial':
    st.title('Ready to try our model?')
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

            with open('financial_inclusion.pkl', 'rb') as clf_inclusion:
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
