import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from utils.preprocessing import features

# Load models
lr = joblib.load("models/linear_regression_model.joblib")
rf = joblib.load("models/random_forest_model.joblib")
xgb = joblib.load("models/xgboost_model.joblib")

lstm_model = load_model("models/lstm_model.keras")
cnn_model = load_model("models/cnn_model.keras")


def predict_single(X):

    lr_pred = lr.predict(X)
    rf_pred = rf.predict(X)
    xgb_pred = xgb.predict(X)

    return lr_pred, rf_pred, xgb_pred


def predict_sequence(X_seq):

    lstm_pred = lstm_model.predict(X_seq)
    cnn_pred = cnn_model.predict(X_seq)

    return lstm_pred.flatten(), cnn_pred.flatten()


def ensemble_prediction(lr_pred, rf_pred, xgb_pred, lstm_pred, cnn_pred):

    pred = (lr_pred + rf_pred + xgb_pred + lstm_pred + cnn_pred) / 5

    return pred


def create_next_row(history, pred_value):

    last = history.iloc[-1].copy()

    new_row = last.copy()

    # predicted demand becomes withdrawals
    new_row["total_withdrawals"] = pred_value

    # assume deposits similar to yesterday
    new_row["total_deposits"] = last["total_deposits"]

    # update previous day cash level
    new_row["previous_day_cash_level"] = last["previous_day_cash_level"] - pred_value

    # increment date features
    next_day = (last["day_of_week"] + 1) % 7
    new_row["day_of_week"] = next_day

    new_row["day"] = last["day"] + 1

    # naive month rollover
    if new_row["day"] > 30:
        new_row["day"] = 1
        new_row["month"] = (last["month"] % 12) + 1

    # rolling features
    temp = history["total_withdrawals"].tolist()
    temp.append(pred_value)

    new_row["rolling_7"] = sum(temp[-7:]) / min(len(temp), 7)
    new_row["rolling_30"] = sum(temp[-30:]) / min(len(temp), 30)

    return new_row


def forecast_next_days(df, days=7):

    history = df.copy().reset_index(drop=True)

    # Ensure unique column names
    history = history.loc[:, ~history.columns.duplicated()]

    predictions = []

    for i in range(days):

        X = history[features].values[-1].reshape(1, -1)

        lr_pred, rf_pred, xgb_pred = predict_single(X)

        seq_data = history[features].values
        seq_data = seq_data[-30:]

        # Pad if fewer than 30 rows
        if seq_data.shape[0] < 30:
            pad = np.repeat(seq_data[0:1], 30 - seq_data.shape[0], axis=0)
            seq_data = np.vstack([pad, seq_data])

        seq_data = seq_data.reshape(1, 30, len(features))

        lstm_pred, cnn_pred = predict_sequence(seq_data)

        final_pred = ensemble_prediction(
            lr_pred, rf_pred, xgb_pred, lstm_pred, cnn_pred
        )

        pred_value = float(final_pred[0])
        predictions.append(pred_value)

        new_row = create_next_row(history, pred_value)

        # Append safely
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return predictions
