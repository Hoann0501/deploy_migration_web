from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

@app.before_first_request
def load_model():
    global model_bundle
    model_bundle = joblib.load("migration_model_stack.pkl")

xgb_model = model_bundle["xgb_model"]
stack_model = model_bundle["stack_model"]
region_encoder = model_bundle["region_encoder"]
province_encoder = model_bundle["province_encoder"]
feature_columns = model_bundle["feature_columns"]

train_df = pd.read_csv("Train.csv")
detailed_df = pd.read_csv("detailed_migration_forecast.csv")

province_names = list(province_encoder.classes_)

@app.route('/')
def index():
    return render_template("index.html", provinces=province_names)

def generate_prediction_with_data(province, year):
    detailed_row = detailed_df[
        (detailed_df["province_name"] == province) & 
        (detailed_df["year"] == year)
    ]
    
    if detailed_row.empty:
        return None, None
    
    row_data = detailed_row.iloc[0]

    base = {
        "year": year,
        "province": province_encoder.transform([province])[0],
        "region": region_encoder.transform([row_data["region_name"]])[0],
        "area": row_data["area"],
        "population": row_data["population"],
        "monthly_income_per_capita": row_data["monthly_income_per_capita"],
        "grdp_per_capita": row_data["grdp_per_capita"],
        "temp_mean": row_data["temp_mean"],
        "total_precip": row_data["total_precip"],
        "precip_hours": row_data["precip_hours"],
        "sunshine_hours": row_data["sunshine_hours"],
        "snowfall": row_data["snowfall"],
        "central_administrated": row_data["central_administrated"],
        "airport": row_data["airport"],
        "maritime_port": row_data["maritime_port"],
        "population_density": row_data["population_density"]
    }

    df_input = pd.DataFrame([base])
    df_encoded = pd.get_dummies(df_input, columns=["region", "province"], drop_first=False)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]

    pred_xgb = xgb_model.predict(df_encoded).reshape(-1, 1)
    pred = stack_model.predict(pred_xgb)[0]

    prediction = float(round(pred, 2))

    return prediction, base

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    province = data['province']
    year = int(data['year'])

    prediction, input_data = generate_prediction_with_data(province, year)
    
    if prediction is None:
        return jsonify({"error": "Không tìm thấy dữ liệu cho tỉnh và năm đã chọn."}), 400

    trend_text = f"""
<div class="trend-text">
    📊 Xu hướng thuận lợi cho di cư 
    <span style="color: {'#28A745' if prediction > 0 else '#007ACC' if prediction == 0 else '#DC3545'}; font-weight: bold;">
        {"vào" if prediction > 0 else "cân bằng" if prediction == 0 else "ra khỏi"}
    </span> 
    khu vực này.
</div>
    """

    explanation = f"""
<div class="data-row">
    <span class="data-label">🏙️ Dân số dự kiến:</span>
    <span class="data-value">{int(input_data['population']):,} người</span>
</div>
<div class="data-row">
    <span class="data-label">💰 GRDP đầu người:</span>
    <span class="data-value">{round(input_data['grdp_per_capita'], 1)} triệu VNĐ/Năm</span>
</div>
<div class="data-row">
    <span class="data-label">💵 Thu nhập TB:</span>
    <span class="data-value">{round(input_data['monthly_income_per_capita'], 2)} triệu VNĐ/Tháng</span>
</div>
<div class="data-row">
    <span class="data-label">👥 Mật độ dân số:</span>
    <span class="data-value">{round(input_data['population_density'], 1)} người/km²</span>
</div>
<div class="data-row">
    <span class="data-label">🌡️ Nhiệt độ TB:</span>
    <span class="data-value">{round(input_data['temp_mean'], 2)}°C</span>
</div>
<div class="data-row">
    <span class="data-label">☀️ Số giờ nắng:</span>
    <span class="data-value">{int(input_data['sunshine_hours']):,} giờ/năm</span>
</div>
    """

    return jsonify({
        "prediction": prediction,
        "trend_text": trend_text.strip(),
        "explanation": explanation.strip()
    })

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Generate predictions for all provinces for map visualization"""
    data = request.json
    year = int(data['year'])
    
    predictions = {}
    for province in province_names:
        prediction, _ = generate_prediction_with_data(province, year)
        if prediction is not None:
            predictions[province] = prediction
    
    return jsonify(predictions)

if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
