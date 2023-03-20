# Inclusión financiera en África
Proyecto de predicción de bancarización de la población en cuatro países de África

# Contexto
En África, la incusión financiera constituye uno de los principales problemas, ya que en una población compuesta de alrededor de 172,19 millones de personas (The World Bank Group , 2020) en 4 países (Kenia, Ruanda, Tanzania y Uganda), solo el 14% de la población adulta, representada por 9,1 millones, tiene este acceso (Zindi).
En este trabajo buscamos predecir quién es más pobrable que posea una cuenta bancaria, lo que potencialmente podría ayudar a entidades bancarias a encontrar potenciales clientes y a la sociedad en general a una mejor calidad de vida.

# Idea de negocio
Nuestro modelo sería de gran utilidad para aquellas instituciones que busquen incentivar la bancarización en la sociedad, como podrían ser entidades bancarias o el Estado de un país.
En ese sentido, el foco para medir la performance de nuestro modelo es predecir con poco error los falsos positivos y con mucho acierto los verdaderos negativos, es decir, capturar con el mayor acierto posible aquellas personas NO bancarizadas.
Entonces, por un lado utilizaremos la métrica de "Precision" para ver qué tan "preciso" es el clasificador al predecir las instancias positivas (bancarizados), y por otro lado, la "Specificity" o "True Negative Rate" para medir la capacidad de detectar los verdaderos negativos (no bancarizados) sobre el total de casos que son negativos.

# Despliegue del modelo
Para el deploy del modelo usamos Streamlit (https://streamlit.io/), un “framework” de Python de código abierto que permite de manera sencilla e integrada desarrollar aplicaciones gracias a la interacción con otras librerías.
El link a la app es el siguiente: https://adilelle1-financial-inclusion-africa-model-q24ypc.streamlit.app/

# Referencias
Financial Inclusion in Africa. Zindi. 2021. Avaliable in: https://zindi.africa/competitions/financial-inclusion-in-africa

World Development Indicators. The World Bank Group. Dec 31th, 2020. Available in: https://datatopics.worldbank.org/world-development-indicators/ 
