from flask import Blueprint, render_template, request
import pandas as pd
import os
from datetime import date
import plotly.express as px
from pathlib import Path

dashboard = Blueprint('dashboard', __name__)
DATA_PATH = (Path(__file__).resolve().parents[1] / "data" / "Data.xlsx")

@dashboard.route('/dashboard', methods=["GET", "POST"])
def index():
    if not DATA_PATH.exists():
        return "<h2>Data file not found.</h2>"

    df = pd.read_excel(DATA_PATH, dtype={"YWT": float})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date", ascending=False)

    # Ensure Date column is datetime and Time is string for grouping
    df["Date"] = pd.to_datetime(df["Date"])
    df['Time']  = df['Time'].astype(str).str.zfill(5)
    df["Shift"] = df["Shift"].astype(str)
    
    shifts = ["D","G"]
    # --- Filters ---
    selected_date = request.form.get("date_filter")
    if not selected_date:
        selected_date = str(date.today())

    selected_shift = request.form.get("shift_filter")

    # Populate dropdown options
    available_dates = sorted(df["Date"].dt.date.unique().tolist())
    available_shifts = sorted(df["Shift"].dropna().unique().tolist())

    filtered_df = df.copy()

    if selected_date:
        selected_date_obj = pd.to_datetime(selected_date).date()
        filtered_df = filtered_df[filtered_df["Date"].dt.date == selected_date_obj]

    if selected_shift:
        filtered_df = filtered_df[filtered_df["Shift"] == selected_shift]

    # Sort by time for line chart
    filtered_df = filtered_df.sort_values("Time")

    chart_html = None
    if not filtered_df.empty:
        fig = px.line(filtered_df, x="Time", y="YWT", title="YWT vs Time", markers=True)

        if selected_shift == "D":
            shift_times = [
                "07:30","08:30","09:30","10:30","11:30","12:30",
                "13:30","14:30","15:30","16:30","17:30","18:30"
            ]
        elif selected_shift == "N":
            shift_times = [
                "19:30","20:30","21:30","22:30","23:30",
                "00:30","01:30","02:30","03:30","04:30","05:30","06:30"
            ]
        else:
            shift_times = sorted(filtered_df['Time'].unique())
        # 1) Turn "Time" into an ordered Categorical…
        filtered_df["Time"] = pd.Categorical(
            filtered_df["Time"],
            categories=shift_times,
            ordered=True,
        )

        # 2) …then sort by it
        filtered_df = filtered_df.sort_values("Time")
        fig = px.line(
                filtered_df,
                x="Time", y="YWT",
                category_orders={"Time": shift_times},
                markers=True,
                title=f"YWT vs Time {selected_shift or 'All'}"
            )
        # Add constant line at YWT = 9.7
        fig.add_shape(
            type="line",
            x0=filtered_df["Time"].min(), x1=filtered_df["Time"].max(),
            y0=9.7, y1=9.7,
            line=dict(color="red", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=filtered_df["Time"].max(),
            y=9.7,
            text="Target YWT = 9.7",
            showarrow=False,
            yshift=10,
            font=dict(color="red")
        )

        fig.update_layout(xaxis_title="Time", yaxis_title="YWT", template="simple_white")
        chart_html = fig.to_html(full_html=False)

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    return render_template(
        'dashboard.html',
        shifts=shifts,                      # ← make sure this is here!
        selected_shift=selected_shift,
        selected_date = selected_date,
        min_date       = min_date,
        max_date       = max_date,
        chart_html     = chart_html
    )