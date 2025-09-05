from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from pathlib import Path
home = Blueprint('home', __name__)

USERS_FILE = (Path(__file__).resolve().parents[1] / "data" / "users.xlsx")


def load_users():
    # ✅ Read when it exists; otherwise return empty with the right columns
    if not USERS_FILE.exists():
        return pd.DataFrame(columns=["role", "username", "password"])

    df = pd.read_excel(USERS_FILE, engine="openpyxl")
    # Normalize column names/values
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"role", "username", "password"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"users.xlsx missing columns: {missing}")

    for col in ["role", "username", "password"]:
        df[col] = df[col].astype(str).str.strip()

    return df

@home.route("/", methods=["GET", "POST"])
def role_selection():
    if request.method == "POST":
        role = (request.form.get("role") or "").strip().lower()
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        # Shift Team – no login required
        if role == "shift":
            return redirect(url_for("dashboard.index"))

        users_df = load_users()
        # normalize role in df too
        users_df["role"] = users_df["role"].str.lower()

        match = users_df[
            (users_df["role"] == role) &
            (users_df["username"].str.casefold() == username.casefold()) &
            (users_df["password"] == password)
        ]

        if not match.empty:
            print(f"✅ Login success: {username} ({role})")
            # 👇 Redirect by role
            if role == "non_shift":
                return redirect(url_for("nonshift.index"))   # make sure you have this blueprint
            elif role == "avp_vp":
                return redirect(url_for("dashboard.index"))
            else:
                return redirect(url_for("dashboard.index"))

        flash("❌ Invalid username or password.")
        print("❌ Login failed")

    return render_template("home.html")
