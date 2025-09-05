from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
import os
from datetime import date

entry = Blueprint('entry', __name__)
file_path = r"C:/Users/ahgua/Documents/Website visual/Data.xlsx"

@entry.route('/entry', methods=["GET", "POST"])
def entry_form():
    day_shift_times = ["7:30", "8:30", "9:30", "10:30", "11:30", "12:30","13:30", "14:30", "15:30", "16:30", "17:30", "18:30"]
    night_shift_times = ["19:30","20:30","21:30","22:30","23:30","00:30","01:30","02:30","03:30","04:30","05:30","06:30",]

    message = None
    shift = time = ywt = selected_date = ""
    success = None  # <-- This fixes your error
    if request.method == "POST":
        shift = request.form.get("shift")
        time = request.form.get("time")
        ywt = request.form.get("ywt")
        selected_date = request.form.get("date")
    
    # Validate fields
        if not shift or not time or not ywt or not selected_date:
            message = "⚠️ Please fill in all fields."
        else:
            try:
                float(ywt)  # validate numeric
                filepath = r"C:/Users/ahgua/Documents/Website visual/Data.xlsx"
                if not os.path.exists(filepath):
                    import openpyxl
                    workbook = openpyxl.Workbook()
                    sheet = workbook.active
                    sheet.append(["Shift", "Date", "Time", "YWT"])
                    workbook.save(filepath)

                df = pd.read_excel(filepath)
                is_duplicate = (
                        (df["Date"].astype(str) == str(pd.to_datetime(selected_date).date())) &
                        (df["Time"] == time) &
                        (df["Shift"] == shift)
                    ).any()
                
                if is_duplicate:
                    message = f"⚠️ An entry already exists for {selected_date} {shift} shift at {time}."
                else:
                    new_row = pd.DataFrame([{
                        "Shift": shift,
                        "Date": pd.to_datetime(selected_date).date(),
                        "Time": time,
                        "YWT": float(ywt)
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_excel(filepath, index=False)
                    time_list = day_shift_times if shift == "D" else night_shift_times
                    try:
                        current_index = time_list.index(time)
                        if current_index < len(time_list) - 1:
                            next_time = time_list[current_index + 1]
                        else:
                            next_time = ""
                    except ValueError:
                        next_time = ""
                    return redirect(url_for("entry_form",
                                            success="1",
                                            shift=shift,
                                            time=next_time,
                                            date=selected_date))
            except ValueError:
                message = "⚠️ YWT must be a number."
    if not selected_date:
        selected_date = date.today().isoformat()
    if request.method == "GET":
        shift = request.args.get("shift") or ""
        time = request.args.get("time") or ""
        selected_date = request.args.get("date") or date.today().isoformat()
        success = request.args.get("success")
    max_date = date.today().isoformat()
    # Then render with these values:
    return render_template(
        "entry.html",
        day_shift_times=day_shift_times,
        night_shift_times=night_shift_times,
        selected_shift=shift,
        selected_time=time,
        selected_date=selected_date,
        message=None,
        success=success,
        ywt="",
        max_date=max_date,          #  <<<<<<<<<<
        min_date="2000-01-01"       #  or whatever lower bound you like
    )