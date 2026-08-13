import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_csv('../data/processed.csv')
    num_cols = ['edad', 'ingresos', 'num_hijos', 'num_tarjetas', 'prestamos']
    cat_cols = ['estado_civil', 'modalidad_pago', 'hipoteca']

    
    # Separamos la variable objetivo
    X = df[num_cols + cat_cols]
    y = df['riesgo']
    
    ratio_desbalance = y.value_counts()[0] / y.value_counts()[1]
    
    # Preprocesamiento numericas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    # Preprocesamiento categoricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Preprocesador completo
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

    # Mejor modelo
    xgb = XGBClassifier(
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.5,
        gamma=1,
        scale_pos_weight=ratio_desbalance,
        random_state=42,
    )

    # Pipeline
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb)]
    )

    # Entrenamos y guardamos el modelo
    xgb_pipeline.fit(X, y)
    joblib.dump(xgb_pipeline, '../models/model_pipeline.joblib')
    print('¡Pipeline entrenado y guardado exitosamente!')

if __name__ == '__main__':
    train()