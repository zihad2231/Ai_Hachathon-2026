from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="UpayPulse Intelligence Engine", version="2.0")

# Model Loading
try:
    bundle = joblib.load("upay_production_bundle.joblib")
    churn_model = bundle["churn_model"]
    nba_model = bundle["nba_model"]
    feature_encoders = bundle["feature_encoders"]
    target_encoder = bundle["target_encoder"]
    feature_names = bundle["feature_names"]
    print("Bundle loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading bundle: {e}")

class CustomerTelemetry(BaseModel):
    Age: int
    Gender: str
    AccountAgeMonths: int
    LifeStageSegment: str
    PreferredCategory: str
    AppOpenFreqWeekly: int
    AvgMonthlyTransactions: int
    AvgTransactionAmount: float
    SupportTicketsCount: int
    CashInVsCashOutRatio: float
    BillPaymentFrequency: int
    FailedTransactionCount: int
    DiscountsRedeemedLastQuarter: int
    AppVersionUpToDate: int
    DaysSinceLastTransaction: int

static_folder = "Static" if os.path.exists("Static") else "static"
app.mount(f"/{static_folder.lower()}", StaticFiles(directory=static_folder), name="static")

@app.get("/")
def serve_index():
    return FileResponse(f"{static_folder}/index.html")

@app.post("/predict")
def predict_customer_risk(data: CustomerTelemetry):
    try:
        input_dict = data.dict()
        df_row = pd.DataFrame([input_dict])

        # Encoders
        for col in ["Gender", "LifeStageSegment", "PreferredCategory"]:
            if col in feature_encoders:
                val = str(df_row[col].iloc[0])
                encoder = feature_encoders[col]
                if val in encoder.classes_:
                    df_row[col] = encoder.transform([val])[0]
                else:
                    df_row[col] = 0

        X_sample = df_row[feature_names]

        # 1. Churn Probability & Risk
        churn_prob = float(churn_model.predict_proba(X_sample)[0][1])
        churn_percent = round(churn_prob * 100, 1)

        if churn_percent >= 65:
            risk_tier = "CRITICAL CHURN RISK"
            color_theme = "red"
        elif churn_percent >= 38:
            risk_tier = "ELEVATED RISK"
            color_theme = "amber"
        else:
            risk_tier = "HEALTHY & RETAINED"
            color_theme = "emerald"

        # 2. Next-Best-Action Prediction
        nba_idx = nba_model.predict(X_sample)[0]
        nba_offer = str(target_encoder.inverse_transform([nba_idx])[0])

        # 3. Dynamic Telemetry Drivers (All 15 Feature impact indicators)
        inactivity_score = min(100, int((data.DaysSinceLastTransaction / 60.0) * 100))
        support_friction = min(100, int((data.SupportTicketsCount / 5.0) * 80 + (data.FailedTransactionCount / 4.0) * 20))
        engagement_decay = max(5, int(100 - (data.AppOpenFreqWeekly * 3.5 + data.AvgMonthlyTransactions * 2.0)))
        app_obsolescence = 80 if data.AppVersionUpToDate == 0 else 15
        monetary_imbalance = min(100, int(abs(1.0 - data.CashInVsCashOutRatio) * 45))

        # 4. Tailored Action Strategy
        if "Cashback" in nba_offer:
            exec_strategy = f"Auto-credit 5% ({nba_offer}) on next transaction. Broadcast push alert with 48h validity."
        elif "Zero-Fee" in nba_offer:
            exec_strategy = "Waive cash-out fee instantly for next 3 cash-out operations to restore transactional velocity."
        elif "Merchant" in nba_offer:
            exec_strategy = "Direct routing to Tier-2 Priority Relationship Desk + SMS voucher for top merchant brands."
        elif "Utility" in nba_offer:
            exec_strategy = "Trigger ৳50 bill payment cashback notification 3 days prior to month-end bill cycles."
        else:
            exec_strategy = "Offer higher annual interest yield boost on wallet balance held over 15 days."

        return {
            "churn_probability": churn_percent,
            "risk_tier": risk_tier,
            "color_theme": color_theme,
            "next_best_action": nba_offer,
            "execution_strategy": exec_strategy,
            "telemetry_drivers": {
                "inactivity_score": inactivity_score,
                "support_friction": min(100, support_friction),
                "engagement_decay": min(100, engagement_decay),
                "app_obsolescence": app_obsolescence,
                "monetary_imbalance": min(100, monetary_imbalance)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))