import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='Risk Model')

# Cargamos el modelo en memoria
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load("models/model_pipeline.joblib")
    except Exception as e:
        return None
pipeline = cargar_modelo()

# Cargamos el dataset en memoria
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/processed.csv')
    except Exception as e:
        return None
df = load_data()

# Inicializamos la variable en el session_state
if 'menu_opcion' not in st.session_state:
    st.session_state['menu_opcion'] = 'Inicio'

def ir_a_evaluacion():
    st.session_state['menu_opcion'] = 'Evaluación de Clientes'

def ir_a_dashboards():
    st.session_state['menu_opcion'] = 'Dashboards'


def inicio():
    st.markdown('# Sistema de Evaluación de Riesgo Crediticio')
    st.markdown('### Plataforma interactiva diseñada para la optimización en la aprobación de créditos y la mitigación de impagos.')
    st.divider()
    st.text('')
    st.text('')
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('#### 🎯 Evaluación de Clientes')
            st.write('Ingresa los datos de un solicitante y obtén un diagnóstico de riesgo en tiempo real')
            st.button('Simular Evaluación', use_container_width=True, on_click=ir_a_evaluacion)

    with col2:
        with st.container(border=True):
            st.markdown('#### 📊 Dashboards')
            st.write('Explora el comportamiento de los clientes según el grupo de riesgo al que pertenecen')
            st.button('Ver Dashboards', use_container_width=True, on_click=ir_a_dashboards)

    
def evaluacion_cliente():
    st.markdown('# 🎯 Evaluación de Clientes')
    st.markdown('### Modelo de Machine Learning para predecir la probabilidad del riesgo de impago.')
    st.divider()
    st.text('')
    
    # Creamos el formulario
    with st.form(key='form_evaluacion_riesgo'):

        col1, col2 = st.columns(2)
        
        # Perfil Demográfico
        with col1:
            st.markdown('##### 👤 Perfil Demográfico')

            # Edad
            user_edad = st.number_input('Edad', min_value=18, value=25, max_value=90, step=1)
            
            # Estado civil
            opciones_estado_civil = {
                        'Soltero': 'Soltero',
                        'Casado': 'Casado',
                        'Divorciado': 'Div/Sep/Viudo',
                        'Separado': 'Div/Sep/Viudo',
                        'Viudo': 'Div/Sep/Viudo'}
            
            seleccion_estado_civil = st.selectbox('Estado civil', list(opciones_estado_civil.keys()))
            user_estado_civil = opciones_estado_civil[seleccion_estado_civil]
            
            # Hijos
            opciones_hijos = {
                    '0': 0,
                    '1': 1,
                    '2': 2,
                    '3': 3,
                    '4 o más':4}
            seleccion_hijos = st.selectbox('Cantidad de hijos', list(opciones_hijos.keys()))
            user_hijos = opciones_hijos[seleccion_hijos]

            # Modalidad de Pago
            user_modalidad_pago = st.selectbox('Modalidad de pago', ['Mensual', 'Semanal'])

        # Perfil Financiero
        with col2:
            st.markdown('##### 💳 Perfil Financiero')

            # Ingresos
            user_ingresos = st.number_input('Ingresos anuales (USD)', 
                            min_value=0, value=40000, step=1000)
            
            # Prestamos
            opciones_prestamos = {
                '0': 0,
                '1': 1,
                '2': 2,
                '3 o más': 3}
            seleccion_prestamos = st.selectbox('Cantidad de créditos activos', 
                                                list(opciones_prestamos.keys()),
                                                index=0)
            user_prestamos = opciones_prestamos[seleccion_prestamos]
                
            # Número de Tarjetas
            opciones_tarjetas = {
                        '0': 0,
                        '1': 1,
                        '2': 2,
                        '3': 3,
                        '4':4,
                        '5':5,
                        '6 o más':6}
            seleccion_tarjetas = st.selectbox('Cantidad de tarjetas', 
                                                list(opciones_tarjetas.keys()),
                                                index=2)
            user_tarjetas = opciones_tarjetas[seleccion_tarjetas]
            
            # Hipoteca
            user_hipoteca = st.selectbox('Posee crédito de hipoteca', ['No','Si'])
            
        # Botón de submit
        submit_button = st.form_submit_button('⚡ Calcular Riesgo', use_container_width=True)

    # Accion del boton
    if submit_button:

        # Organizamos los inputs en un df
        datos_cliente = pd.DataFrame([{
            'edad': user_edad,
            'estado_civil': user_estado_civil,
            'num_hijos': user_hijos,
            'ingresos': user_ingresos,
            'prestamos': user_prestamos,
            'num_tarjetas': user_tarjetas,
            'modalidad_pago': user_modalidad_pago,
            'hipoteca': user_hipoteca,
            }])
    
        # Realizamos prediccion
        def predecir_riesgo(pipeline, df_cliente):
            prob_impago = pipeline.predict_proba(df_cliente)[0][1]
            return prob_impago

        prob_impago = predecir_riesgo(pipeline, datos_cliente)
        
        
        # Ajustamos el score para datos que violen políticas básicas de crédito
    
        # Regla 1: Ingresos extremadamente bajos
        if user_ingresos < 8000:
            prob_impago = max(prob_impago, 0.8)

        # Regla 2: Edad avanzada con ingresos bajos
        elif user_edad > 65 and user_ingresos < 15000:
            prob_impago = max(prob_impago, 0.7)

        # Regla 3: Muchos prestamos activos, con múltiples tarjetas e hipoteca
        elif user_prestamos == 3 and user_tarjetas >= 3 and user_hipoteca == 'Si':
            prob_impago = max(prob_impago, 0.65)
        
        porcentaje_impago = prob_impago * 100
        
        st.divider()
            
        # Mostramos resultado
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric(label='Probabilidad de Impago',value=f'{porcentaje_impago:.1f}%')

        if porcentaje_impago < 30.0:
            with col_m2:
                st.metric(label='Nivel de Riesgo', value='Bajo 🟢')
            st.success('**Estatus: Solicitud Aprobada.** El perfil presenta un riesgo financiero mínimo')

        elif 30.0 <= porcentaje_impago < 50.0:
            with col_m2:
                st.metric(label='Nivel de Riesgo', value='Moderado 🟡')
            st.warning('**Estatus: Requiere Revisión.** El perfil presenta un riesgo moderado.')

        else:
            with col_m2:
                st.metric(label='Nivel de Riesgo', value='Alto 🔴')
            st.error('**Estatus: Solicitud Rechazada.** La probabilidad de impago supera el umbral de tolerancia institucional.')
            
    
def dashboards():
    st.markdown('# 📊 Dashboards')
    st.divider()

    # Grafico 1 ---------------------------------------
    num_cols = {'Edad':'edad',
                'Ingresos':'ingresos',
                }
    
    col1, _, col2 = st.columns([3, 0.5, 1.5])
    
    with col1:
        seleccion_col_num = st.selectbox('Columna',list(num_cols.keys()))
        user_col_num = num_cols[seleccion_col_num]
        
    with col2:
        segmentar_riesgo_num = st.radio('Segmentar por riesgo',['No', 'Sí'],horizontal=True, index=0)
        
    if segmentar_riesgo_num == 'No':
        color_var = None
        barmode_var = None
        paleta_colores = px.colors.qualitative.Plotly
        titulo_grafico = f'Histograma de {seleccion_col_num}'
        
    else:
        color_var = 'riesgo'
        barmode_var = 'overlay'
        paleta_colores = px.colors.qualitative.Set2 
        titulo_grafico = f'Distribución de {seleccion_col_num} segmentada por Riesgo'
    
    with st.container():
        fig = px.histogram(
            data_frame=df, 
            x=user_col_num, 
            nbins=30,
            color=color_var,
            barmode=barmode_var,
            title=titulo_grafico,
            color_discrete_sequence=paleta_colores,
            labels={user_col_num: seleccion_col_num,
                    'count': 'count',
                    'riesgo': 'Riesgo'})
        
        st.plotly_chart(fig, use_container_width=True)
            
    
    st.divider()
    
    # Grafico 2 ---------------------------------------
    cat_cols = {'Prestamos':'prestamos',
                'Número de Tarjetas':'num_tarjetas',
                'Número de Hijos':'num_hijos',
                'Estado Civil':'estado_civil',
                'Modalidad de Pago':'modalidad_pago',
                'Hipoteca':'hipoteca'
                }
    
    col_select, _, col_radio = st.columns([3, 0.5, 1.5])
    
    with col_select:
        seleccion_col_cat = st.selectbox('Columna',list(cat_cols.keys()))
        user_col_cat = cat_cols[seleccion_col_cat]
        
    with col_radio:
        segmentar_riesgo = st.radio('Segmentar por riesgo',['No', 'Sí'],horizontal=True, index=1)
    
    if segmentar_riesgo == 'No':
        df_temp = df.groupby(user_col_cat).size().reset_index(name='count')
    
        color_var = None
        barmode_var = None
        titulo_grafico = f'Conteo general de {seleccion_col_cat}'
        paleta_colores = px.colors.qualitative.Plotly
        
    else:
        df_temp = df.groupby([user_col_cat, 'riesgo']).size().reset_index(name='count')
        df_temp['riesgo'] = df_temp['riesgo'].astype(str)
        
        color_var = 'riesgo'
        barmode_var = 'group'
        titulo_grafico = f'Conteo de {seleccion_col_cat} segmentado por Riesgo'
        paleta_colores = px.colors.qualitative.Set2
        

    with st.container():
        fig = px.bar(data_frame=df_temp, 
                    x=user_col_cat, 
                    y='count',
                    color=color_var,             
                    barmode=barmode_var,         
                    title=titulo_grafico,
                    labels={user_col_cat: seleccion_col_cat, 
                            'count': 'count', 
                            'riesgo': 'Riesgo'},
                    color_discrete_sequence=paleta_colores
                    )
        
        orden_eje = sorted(df_temp[user_col_cat].unique())
        fig.update_xaxes(type='category', 
                        categoryorder='array', 
                        categoryarray=orden_eje
                        )
        
        st.plotly_chart(fig, use_container_width=True)    
                

menu = st.sidebar.selectbox('Menú', ['Inicio', 'Evaluación de Clientes', 'Dashboards'], key='menu_opcion')

if menu == 'Inicio':
    inicio()
    
elif menu == 'Evaluación de Clientes':
    evaluacion_cliente()
    
elif menu == 'Dashboards':
    dashboards()