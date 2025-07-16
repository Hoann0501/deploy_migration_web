from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model_bundle = joblib.load("migration_model_stack.pkl")
xgb_model = model_bundle["xgb_model"]
stack_model = model_bundle["stack_model"]
region_encoder = model_bundle["region_encoder"]
province_encoder = model_bundle["province_encoder"]
feature_columns = model_bundle["feature_columns"]

train_df = pd.read_csv("Train.csv")

province_names = list(province_encoder.classes_)

@app.route('/')
def index():
    return render_template("index.html", provinces=province_names)

def generate_prediction(province, year):
    region_row = train_df[train_df["province"] == province]
    if region_row.empty:
        return None
    region = region_row.iloc[0]["region"]

    base = {
        "year": year,
        "province": province_encoder.transform([province])[0],
        "region": region_encoder.transform([region])[0],
        "area": 1000,
        "population": 1_000_000 * (1.015) ** (year - 2024),
        "monthly_income_per_capita": 5 * (1.05) ** (year - 2024),
        "grdp_per_capita": 40 * (1.045) ** (year - 2024),
        "temp_mean": 25 + 0.015 * (year - 2024),
        "total_precip": 2000 * (1.005) ** (year - 2024),
        "precip_hours": 2000 * (1.004) ** (year - 2024),
        "sunshine_hours": 180000 * (1.002) ** (year - 2024),
        "snowfall": 0,
        "central_administrated": 0,
        "airport": 1,
        "maritime_port": 0
    }
    base["population_density"] = base["population"] / base["area"]

    df_input = pd.DataFrame([base])
    df_encoded = pd.get_dummies(df_input, columns=["region", "province"], drop_first=False)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]

    pred_xgb = xgb_model.predict(df_encoded).reshape(-1, 1)
    pred = stack_model.predict(pred_xgb)[0]

    return float(round(pred, 2))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    province = data['province']
    year = int(data['year'])

    prediction = generate_prediction(province, year)
    if prediction is None:
        return jsonify({"error": "Không tìm thấy tỉnh đã chọn."}), 400

    region_row = train_df[train_df["province"] == province]
    region = region_row.iloc[0]["region"]

    base = {
        "year": year,
        "province": province_encoder.transform([province])[0],
        "region": region_encoder.transform([region])[0],
        "area": 1000,
        "population": 1_000_000 * (1.015) ** (year - 2024),
        "monthly_income_per_capita": 5 * (1.05) ** (year - 2024),
        "grdp_per_capita": 40 * (1.045) ** (year - 2024),
        "temp_mean": 25 + 0.015 * (year - 2024),
        "total_precip": 2000 * (1.005) ** (year - 2024),
        "precip_hours": 2000 * (1.004) ** (year - 2024),
        "sunshine_hours": 180000 * (1.002) ** (year - 2024),
        "snowfall": 0,
        "central_administrated": 0,
        "airport": 1,
        "maritime_port": 0
    }
    base["population_density"] = base["population"] / base["area"]

    explanation = f"""
        Dân số dự kiến: {int(base['population']):,} người  
GRDP đầu người: {round(base['grdp_per_capita'], 1)} triệu VNĐ  
Thu nhập TB: {round(base['monthly_income_per_capita'], 2)} triệu VNĐ  
Mật độ dân số: {round(base['population_density'], 1)} người/km²  
Nhiệt độ TB: {round(base['temp_mean'], 2)}°C, Nắng: {int(base['sunshine_hours'])} giờ/năm  
Các đặc trưng trên có xu hướng thuận lợi cho di cư {"vào" if prediction > 0 else "ra"} tại khu vực này.
    """

    return jsonify({
        "prediction": prediction,
        "explanation": explanation.strip()
    })

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Generate predictions for all provinces for map visualization"""
    data = request.json
    year = int(data['year'])
    
    predictions = {}
    for province in province_names:
        prediction = generate_prediction(province, year)
        if prediction is not None:
            predictions[province] = prediction
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run()