import pandas as pd

# Leer el archivo CSV
file_path = 'models/XGB/10CV/XGB_10CV_MAP@3-035011/XGB_10CV_MAP@3-035011_submission.csv'
df = pd.read_csv(file_path)

print(f'Archivo original:')
print(f'  - Filas: {len(df)}')
print(f'  - ID mínimo: {df["id"].min()}')
print(f'  - ID máximo: {df["id"].max()}')

# Corregir la columna ID para que vaya de 750000 a 999999
df['id'] = range(750000, 750000 + len(df))

print(f'Archivo corregido:')
print(f'  - Filas: {len(df)}')
print(f'  - ID mínimo: {df["id"].min()}')
print(f'  - ID máximo: {df["id"].max()}')

# Guardar el archivo corregido
df.to_csv(file_path, index=False)
print(f'✅ Archivo guardado con IDs corregidos: {file_path}')

# Mostrar algunas muestras para verificar
print('\nPrimeras 5 filas:')
print(df.head())

print('\nÚltimas 5 filas:')
print(df.tail())
