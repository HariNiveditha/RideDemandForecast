import json, pathlib
p = pathlib.Path(r'C:\Users\vamsh\Desktop\RideDemandForecast\training.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if 'model_df["rolling_mean_24"]' in src or 'dropna(subset=["lag_1"' in src or 'hour_of_day' in src and 'model_df' in src:
        print('CELL', i)
        print(src)
        print('\n---\n')
