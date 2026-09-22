import json, pathlib
p = pathlib.Path(r'C:\Users\vamsh\Desktop\RideDemandForecast\training.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if 'train_test_split' in src or 'X_test' in src or 'y_test' in src or 'rf_model' in src:
        print('CELL', i)
        print(src[:5000])
        print('\n---\n')
