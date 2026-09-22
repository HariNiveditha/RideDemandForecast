 # RideCast frontend

 This is the React/Vite UI for the ride demand forecasting project. It preserves the notebooks in the repository and expects a Python inference service at `/api/predict`.

 ## Run

 ```bash
 npm install
 npm run dev
 ```

 ## Inference contract

 The forecast form sends a `POST /api/predict` request with:

 ```json
 {
   "pickup_location": "...",
   "pickup_datetime": "2021-06-30T17:00",
   "weather": {
     "temp": 22.1,
     "humidity": 60,
     "precip": 0
   }
 }
 ```

 The service should load the Random Forest trained with the weather and lag feature set from `training.ipynb`. Its inference layer must resolve `lag_1`, `lag_24`, `lag_168`, and `rolling_mean_24` from the uploaded ride history before calling the model. The frontend intentionally displays an unavailable state until that service and the parquet/CSV files are present.
