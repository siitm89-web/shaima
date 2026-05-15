
import base64
import io
import os
from datetime import datetime, timedelta

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request

matplotlib.use("Agg")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "lgbm_review_count_model.txt"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "merged_weather_reviews.csv"
)

loaded_model = lgb.Booster(model_file=MODEL_FILE)

features = [
    "tavg", "tmin", "tmax",
    "prcp", "wspd", "pres",
    "lag1", "lag2", "lag3",
    "lag7", "lag14", "lag30",
    "rolling_mean_7", "rolling_std_7", "rolling_max_7",
    "rolling_mean_14", "rolling_std_14", "rolling_max_14",
    "rolling_mean_30", "rolling_std_30", "rolling_max_30"
]

DEFAULT_LATITUDE = 18.2164
DEFAULT_LONGITUDE = 42.5053


def get_weather_forecast_dataframe(
        start_date,
        end_date,
        latitude,
        longitude
):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "wind_speed_10m_max",
            "surface_pressure_mean"
        ],
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    weather_forecast_df = pd.DataFrame({
        "time": pd.to_datetime(data["daily"]["time"]),
        "tavg": data["daily"]["temperature_2m_mean"],
        "tmin": data["daily"]["temperature_2m_min"],
        "tmax": data["daily"]["temperature_2m_max"],
        "prcp": data["daily"]["precipitation_sum"],
        "wspd": data["daily"]["wind_speed_10m_max"],
        "pres": data["daily"]["surface_pressure_mean"]
    })

    return weather_forecast_df


def prepare_historical_data(csv_path=CSV_PATH):

    merged_data = pd.read_csv(csv_path)

    merged_data["time"] = pd.to_datetime(
        merged_data["time"]
    )

    merged_data = merged_data.sort_values("time")

    if len(merged_data) < 30:
        raise ValueError(
            "Dataset must contain at least 30 rows"
        )

    historical_data = merged_data.copy()

    historical_data["lag1"] = historical_data["review_count"].shift(1)
    historical_data["lag2"] = historical_data["review_count"].shift(2)
    historical_data["lag3"] = historical_data["review_count"].shift(3)
    historical_data["lag7"] = historical_data["review_count"].shift(7)
    historical_data["lag14"] = historical_data["review_count"].shift(14)
    historical_data["lag30"] = historical_data["review_count"].shift(30)

    historical_data["rolling_mean_7"] = (
        historical_data["review_count"].rolling(7).mean()
    )

    historical_data["rolling_std_7"] = (
        historical_data["review_count"].rolling(7).std()
    )

    historical_data["rolling_max_7"] = (
        historical_data["review_count"].rolling(7).max()
    )

    historical_data["rolling_mean_14"] = (
        historical_data["review_count"].rolling(14).mean()
    )

    historical_data["rolling_std_14"] = (
        historical_data["review_count"].rolling(14).std()
    )

    historical_data["rolling_max_14"] = (
        historical_data["review_count"].rolling(14).max()
    )

    historical_data["rolling_mean_30"] = (
        historical_data["review_count"].rolling(30).mean()
    )

    historical_data["rolling_std_30"] = (
        historical_data["review_count"].rolling(30).std()
    )

    historical_data["rolling_max_30"] = (
        historical_data["review_count"].rolling(30).max()
    )

    last_dataset_date = merged_data["time"].max()

    return (
        merged_data,
        historical_data,
        last_dataset_date
    )


def forecast_until_date(
        need_to_date,
        weather_forecast_df
):

    merged_data, historical_data, last_dataset_date = (
        prepare_historical_data()
    )

    weather_forecast_df["time"] = pd.to_datetime(
        weather_forecast_df["time"]
    )

    need_to_date = pd.to_datetime(need_to_date)

    forecast_window = weather_forecast_df[
        (weather_forecast_df["time"] > last_dataset_date) &
        (weather_forecast_df["time"] <= need_to_date)
    ].copy()

    forecast_window = forecast_window.sort_values("time")

    if forecast_window.empty:
        return (
            merged_data,
            pd.DataFrame([]),
            last_dataset_date
        )

    last_values = historical_data.copy()

    predictions = []

    for _, weather_row in forecast_window.iterrows():

        last_row = last_values.iloc[-1]

        input_row = pd.DataFrame([{

            "tavg": weather_row["tavg"],
            "tmin": weather_row["tmin"],
            "tmax": weather_row["tmax"],
            "prcp": weather_row["prcp"],
            "wspd": weather_row["wspd"],
            "pres": weather_row["pres"],

            "lag1": last_values["review_count"].iloc[-1],
"lag2": last_values["review_count"].iloc[-2],
"lag3": last_values["review_count"].iloc[-3],
"lag7": last_values["review_count"].iloc[-7],
"lag14": last_values["review_count"].iloc[-14],
"lag30": last_values["review_count"].iloc[-30],
            
            "rolling_mean_7": (
                last_values["review_count"]
                .tail(7)
                .mean()
            ),

            "rolling_std_7": (
                last_values["review_count"]
                .tail(7)
                .std()
            ),

            "rolling_max_7": (
                last_values["review_count"]
                .tail(7)
                .max()
            ),

            "rolling_mean_14": (
                last_values["review_count"]
                .tail(14)
                .mean()
            ),

            "rolling_std_14": (
                last_values["review_count"]
                .tail(14)
                .std()
            ),

            "rolling_max_14": (
                last_values["review_count"]
                .tail(14)
                .max()
            ),

            "rolling_mean_30": (
                last_values["review_count"]
                .tail(30)
                .mean()
            ),

            "rolling_std_30": (
                last_values["review_count"]
                .tail(30)
                .std()
            ),

            "rolling_max_30": (
                last_values["review_count"]
                .tail(30)
                .max()
            )

        }])

        input_row = input_row[features]

        pred_log = loaded_model.predict(input_row)

        pred = np.expm1(pred_log)[0]
        print("Prediction:", pred)

        predictions.append({
            "time": weather_row["time"],
            "prediction": pred
        })

        new_row = last_row.copy()

        new_row["review_count"] = pred

        last_values = pd.concat([
            last_values,
            new_row.to_frame().T
        ])

    pred_df = pd.DataFrame(predictions)

    return (
        merged_data,
        pred_df,
        last_dataset_date
    )


def render_plot(
        merged_data,
        pred_df,
        last_dataset_date
):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        merged_data["time"],
        merged_data["review_count"],
        label="Historical Review Count",
        color="#0d6efd",
        linewidth=2
    )

    ax.plot(
        pred_df["time"],
        pred_df["prediction"],
        label="Predicted Review Count",
        color="#dc3545",
        marker="o",
        linestyle="--",
        linewidth=2
    )

    ax.axvline(
        x=last_dataset_date,
        linestyle="--",
        color="#6c757d",
        alpha=0.8
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Review Count")

    ax.set_title(
        "Historical and Predicted Review Counts"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    buf = io.BytesIO()

    fig.tight_layout()

    fig.savefig(
        buf,
        format="png",
        dpi=120
    )

    plt.close(fig)

    buf.seek(0)

    return base64.b64encode(
        buf.read()
    ).decode("utf-8")


def table_records_from_weather(df):

    return [
        {
            "time": row["time"].strftime("%Y-%m-%d"),
            "tavg": f"{row['tavg']:.1f}",
            "tmin": f"{row['tmin']:.1f}",
            "tmax": f"{row['tmax']:.1f}",
            "prcp": f"{row['prcp']:.2f}",
            "wspd": f"{row['wspd']:.2f}",
            "pres": f"{row['pres']:.1f}"
        }

        for _, row in df.iterrows()
    ]


def table_records_from_forecast(df):

    return [
        {
            "time": row["time"].strftime("%Y-%m-%d"),
            "prediction": f"{row['prediction']:.2f}"
        }

        for _, row in df.iterrows()
    ]


@app.route("/")
def home():

    return render_template("home.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    today = datetime.now().strftime("%Y-%m-%d")

    default_date = (
        datetime.now() + timedelta(days=7)
    ).strftime("%Y-%m-%d")

    prediction_date = default_date

    weather_rows = []

    forecast_rows = []

    diagnostic_info = {}

    plot_image = None

    error = None

    if request.method == "POST":

        prediction_date = request.form.get(
            "prediction_date",
            default_date
        )

        try:

            weather_forecast_df = (
                get_weather_forecast_dataframe(
                    start_date=today,
                    end_date=prediction_date,
                    latitude=DEFAULT_LATITUDE,
                    longitude=DEFAULT_LONGITUDE
                )
            )

            merged_data, forecast_df, last_dataset_date = (
                forecast_until_date(
                    need_to_date=prediction_date,
                    weather_forecast_df=weather_forecast_df
                )
            )

            if forecast_df.empty:

                error = (
                    "No future forecast data available"
                )

            else:

                plot_image = render_plot(
                    merged_data,
                    forecast_df,
                    last_dataset_date
                )

                weather_rows = (
                    table_records_from_weather(
                        weather_forecast_df
                    )
                )

                forecast_rows = (
                    table_records_from_forecast(
                        forecast_df
                    )
                )

                diagnostic_info = {

                    "Last dataset date":
                        last_dataset_date.strftime("%Y-%m-%d"),

                    "Historical mean":
                        f"{merged_data['review_count'].mean():.2f}",

                    "Forecast mean":
                        f"{forecast_df['prediction'].mean():.2f}",

                    "Forecast max":
                        f"{forecast_df['prediction'].max():.2f}",

                    "Forecast min":
                        f"{forecast_df['prediction'].min():.2f}"
                }

        except Exception as exc:

            error = str(exc)

    return render_template(
        "dashboard.html",

        prediction_date=prediction_date,

        min_date=today,

        weather_rows=weather_rows,

        forecast_rows=forecast_rows,

        diagnostic_info=diagnostic_info,

        plot_image=plot_image,

        error=error
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5001
    )

