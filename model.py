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
from plotly.subplots import make_subplots
import plotly.graph_objects as go




#1. Import
df = pd.read_csv('Train.csv')
df.drop(columns=['uniqueid', 'year'],inplace=True)

#2. Titulo de pagina
st.set_page_config(page_title="Financial inclusion prediction App")

#3. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Data visualization', 'Model backstage', 'Model trial'],
    )


#####################################################################################################################################


# Pagina 1 = Home
if selected == 'Home':
    st.title('Financial inclusion in Africa')
    st.write('Predict who in Africa is most likely to have a bank account')


    st.header('Problemática y objetivos')
    st.write('La inclusión financiera refiere al acceso que tienen las personas y las empresas a diversos productos y servicios financieros útiles y asequibles que atienden sus necesidades. Representa una preocupación global, ya que se considera como elemento facilitador para reducir la pobreza extrema y promover el crecimiento y desarrollo económico.')
    st.write('El acceso a cuentas bancarias impacta tanto en el desarrollo humano como en el ámbito económico, ya que permite a los hogares ahorrar, realizar pagos, acceder a créditos, financiamiento, entre otros; al mismo tiempo que ayuda a las empresas a aumentar su solvencia crediticia y mejorar su acceso a préstamos, seguros y servicios relacionados.')
    st.write('En África, la incusión financiera constituye uno de los principales problemas, ya que en una población compuesta de alrededor de 172,19 millones de personas (The World Bank Group , 2020) en 4 países (Kenia, Ruanda, Tanzania y Uganda), solo el 14% de la población adulta, representada por 9,1 millones, tiene este acceso (Zindi).')
    st.write('Por esta razón, la Nueva Alianza para el Desarrollo de África (NEPAD) ha involucrado a representantes del continente para encontrar soluciones que podrían mejorar la inclusión financiera y el bienestar de las personas que viven en África, utilizando el modelo empresarial cooperativo.')
    st.write('En este trabajo buscamos predecir quién es más pobrable que tenga una cuenta bancaria, lo que potencialmente podría ayudar a entidades bancarias a encontrar potenciales clientes.')
    
    st.header('Dataset')
    st.write('El conjunto de datos utilizado, ha sido extraído de Zindi, una red profesional para científicos de datos en África; constituyen los resultados de las encuestas de Finscope de 2016 a 2018.')
    st.write("[Zindi website](https://zindi.africa/competitions/financial-inclusion-in-africa)")

    st.write('Su contenido principal está relacionado con la información demográfica y servicios financieros utilizados por aproximadamente 23.500 personas en África, correspondientes a 4 países: Kenia, Ruanda, Tanzania y Uganda.')
    st.write("A continuación podemos ver cómo se compone el set de datos")
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

    st.header('Idea de negocio')
    st.write('Nuestro modelo sería de gran utilidad para aquellas instituciones que busquen incentivar la bancarización en la sociedad, como podrían ser entidades bancarias o el Estado de un país.')
    st.write('En ese sentido, el foco para medir la performance de nuestro modelo es predecir con poco error los falsos positivos y con mucho acierto los verdaderos negativos, es decir, capturar con el mayor acierto posible aquellas personas NO bancarizadas.')
    st.write('Entonces, por un lado utilizaremos la métrica de "Precision" para ver qué tan "preciso" es el clasificador al predecir las instancias positivas (bancarizados), y por otro lado, la "Specificity" o "True Negative Rate" para medir la capacidad de detectar los verdaderos negativos (no bancarizados) sobre el total de casos que son negativos.')
    


#####################################################################################################################################


# Pagina 2 = Graficos
elif selected == 'Data visualization':
    st.title('Data visualization')

    #histplot
    col_histplot = st.sidebar.selectbox('Columna - Histplot',['country','location_type', 'age_of_respondent','household_size','relationship_with_head','marital_status','education_level','job_type'])
    def graf_hist():
        fig = px.histogram(df, x= col_histplot, color=col_histplot, color_discrete_sequence=px.colors.qualitative.Set2).update_xaxes(categoryorder='total descending')
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            bargap= 0.2,
            yaxis = dict(tickfont = dict(size=18)),
            xaxis = dict(tickfont = dict(size=18)),
            showlegend=False,
            )
        st.plotly_chart(fig)


    # pie chart
    col_piechart = st.sidebar.selectbox('Columna - Pie chart',['bank_account','gender_of_respondent','cellphone_access'])
    def graf_pie():
        fig = px.pie(df, names=col_piechart, color=col_piechart, color_discrete_sequence=px.colors.qualitative.Set2, hole=.5)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            legend=dict(font=dict(size= 18))
        )
        st.plotly_chart(fig)


    # Histplot by feature
    col_hist_by_feat = st.sidebar.selectbox('Columna - Histplot cuenta bancaria por variable',['country','location_type', 'household_size','relationship_with_head','marital_status','education_level','job_type'])
    def graf_hist_by_feature():
        fig = px.histogram(df, x=['bank_account'], color= col_hist_by_feat, barmode='group',  color_discrete_sequence=px.colors.qualitative.Set2).update_xaxes(categoryorder='total descending',)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            bargap= 0.2,
            legend=dict(font=dict(size= 18)),
            xaxis = dict(showticklabels = True, tickfont = dict(size = 18))
            )
        st.plotly_chart(fig)


    # boxplot
    col_box_plot = st.sidebar.selectbox('Columna - Boxplot',['age_of_respondent','household_size'])
    def boxplot():
        fig = px.box(df, x=col_box_plot, color='bank_account')
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            legend=dict(font=dict(size= 18)),
            xaxis = dict(showticklabels = True, tickfont = dict(size = 16))
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
            height=600,
            bargap=0.2,
            yaxis = dict(tickfont = dict(size=18)),
            xaxis = dict(tickfont = dict(size=18)),
            showlegend=False
            )
        st.plotly_chart(fig)

    if __name__ == '__main__':
        st.header('Distribución')
        graf_hist()
        st.header('Pie chart')
        graf_pie()
        st.header('Distribución de cuenta bancaria por feature')
        graf_hist_by_feature()
        st.header('Boxplot')
        boxplot()
        st.header('Heatmap')
        heatmap_plot()
        
        st.write('---')
        st.header('Conclusiones del análisis')
        st.markdown('Al realizar el análisis de distribuciones y correlación, fue posible observar que:')
        st.markdown('- Kenya es el país con mayor registro de bancarización.')
        st.markdown('- Existe una diferencia sustancial entre el tipo de locación (zona rural/urbana) de la persona cuando se trata de no tener acceso al banco. Sin embargo, ésto no se repite para quienes sí tienen cuenta.')
        st.markdown('- Cuando se trata del género de la persona encuestada, se observa un comportamiento similar al susodicho.')
        st.markdown('- La edad media de las personas bancarizadas es de 40 años, la de aquellas personas que no poseen cuenta, es de 39; la mediana, es de 36 y 35.')
        st.markdown('- Dentro de las variables, la que tiene mayor correlación con la columna target, es la que refiere al acceso de teléfono celular, con un valor de 0.21; hecho que se explicita al observar la distribución de personas con acceso a celular en relación a estar bancarizadas.')
        st.write('---')

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)


#####################################################################################################################################


# Pagina 3 = Comparación de modelos
elif selected == 'Model backstage':
    def model_backstage():
        st.title('Building a classification model')
        st.write('Luego de entender los datos con los que nos encontramos, para avanzar en el armado del modelo tuvimos que sobreponernos a diferentes cuestiones que surgieron a lo largo de todo el trbaajo.')
        st.write('En esta sección repasaremos estas cuestiones más a detalle para explicar como logramos construir nuestro modelo.')

        st.header('1. Preprocesamiento')
        st.write('El primer paso para crear el modelo fue el de crear un paso del Pipeline para transformar los datos.')
        st.write('Nos encontramos con un dataset con un trabajo de preprocesamiento ya realizado, sin datos faltantes.')
        st.write('Sin embargo, al tener columnas categóricas y numéricas fue necesario realizar un paso de preprocesamiento diferente para cada tipo de dato.')
        st.write('Es por eso que fue necesario utilizar la clase ColumnTransformer de la librería Sklearn, con dos pipelines dentro.')
        st.subheader('ColumnTransformer:')
        st.image('columntransformer.png')

        
        st.header('2. Modelo a elegir')
        st.write('En primer lugar decidimos probar por separado la performance de cada clasificador y hacer una búsqueda de los parámetros óptimos de cada uno.')
        
        def print_model_comparison():
            comparacion_modelos_data = pd.read_csv('comparacion_modelos.csv', index_col='Unnamed: 0')
            comparacion_modelos_data_sorted = comparacion_modelos_data.sort_values(by=['specificity'], ascending=False)
            return st.dataframe(comparacion_modelos_data_sorted)
        print_model_comparison()

        st.write('Vemos que todos los clasificadores obtuvieron resultados muy similares en cuanto al accuracy score, sin embargo, Random Forest Classifier es quien tiene mejores métricas de precisión y especificidad.')
        st.write('Entonces, para nuestro modelo vamos a elegir ese clasificador.')


        st.header('3. Modelo final con eliminación de features')
        st.write('En el paso anterior definimos el modelo a utilizar, y usando GridSearch, sus hiperparámetros ideales.')
        
        st.write('A partir de eso, agregamos a la búsqueda la cantidad de variables ideales para el modelo usando la clase RFE de la librería Sklearn.')
        
        st.write('Los resultados del modelo de clasificación fueron los siguientes:')
        def print_model_scores():
            moodel_scores_data = pd.read_csv('model_scores.csv', index_col='Unnamed: 0')
            return st.dataframe(moodel_scores_data)
        print_model_scores()

        st.markdown('**Recordamos la hipótesis nula del modelo:**')
        st.markdown(f'No bancarizados: **{round(df.bank_account.value_counts(normalize=True)[0]*100, 2)}%**')
        st.markdown(f'Bancarizados: **{round(df.bank_account.value_counts(normalize=True)[1]*100, 2)}%**')

        st.header('Matriz de confusión:')
        st.image('conf_matrix.png')

        st.header('Curva ROC-AUC:')
        st.image('roc_auc.png')


        st.header('4. Desbalanceo de variable objetivo')
        st.write('Distribución de clases de Bank Account:')
        fig = go.Figure()
        fig.add_trace(go.Pie(labels = df['bank_account'], hole = 0.6,))
        st.plotly_chart(fig)

        st.write('Al encontrarnos con nuestra variable objetivo fuertemente desbalanceada decidimos probar usando técnicas de balanceo de clases:')
        st.subheader('- Under sampling')
        st.write('Usamos la clase RandomUnderSampler de la librería Sklearn para realizar un resampleo y entrenar el modelo con un subset de datos descartando casos de la clase mayoritaria.')
        st.write('El resultado fue el siguiente:')
        st.dataframe(pd.read_csv('model_us_scores.csv', index_col='Unnamed: 0'))
        st.image('conf_matrix_us.png')

        
        st.subheader('- Over sampling')
        st.write('Usamos la clase RandomOverSampler de la librería Sklearn para realizar nuevamente un resampleo, pero esta vez aumentando la representación de la clase minoritaria.')
        st.write('El resultado fue el siguiente:')
        st.dataframe(pd.read_csv('model_os_scores.csv', index_col='Unnamed: 0'))
        st.image('conf_matrix_os.png')

        
    if __name__ == '__main__':
        model_backstage()


#####################################################################################################################################


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
                st.markdown('<h4 style="text-align: center; color: Green">El individuo se encuentra bancarizado</h4>',unsafe_allow_html=True)
                st.write('---')
            else:
                st.write('---')
                st.markdown('<h4 style="text-align: center; color: Red">El individuo no se encuentra bancarizado</h4>', unsafe_allow_html=True, )
                st.write('---')

    if __name__ == '__main__':
        print_results()
