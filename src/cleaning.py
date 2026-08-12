import pandas as pd

df = pd.read_csv('data/raw.csv')
df.columns = df.columns.str.lower()
df.genero = df.genero.map({'f':'F','m':'M'})
df.genero = df.genero.fillna(df.genero.mode()[0])
df.estado_civil = df.estado_civil.map({'married':'Casado','Married':'Casado', 'single':'Soltero', 'divsepwid':'Div/Sep/Viudo', 'div':'Div/Sep/Viudo'})
df.modalidad_pago = df.modalidad_pago.map({'monthly':'Mensual', 'weekly':'Semanal'})
df.hipoteca = df.hipoteca.map({'y':'Si','n':'No'})
df.riesgo = df.riesgo.map({'F':0, 'V':1})
df.to_csv('data/processed.csv', index=False)
print('¡Archivo processed.csv creado de manera exitosa!')