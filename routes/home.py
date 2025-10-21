from flask import Blueprint, render_template, request, flash
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

home = Blueprint('home', __name__)

USERS_FILE = (Path(__file__).resolve().parents[1] / "data" / "users.xlsx")

def load_users():
    if not USERS_FILE.exists():
        # If the file doesn't exist, return an empty table with expected columns
        return pd.DataFrame(columns=["role", "username", "password"])
    df = pd.read_excel(USERS_FILE, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["role", "username", "password"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""  # ensure the column exists
    return df

@home.route("/", methods=["GET", "POST"])
def role_selection():
    if request.method == "POST":
        role = (request.form.get("role") or "").strip().lower()
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        users_df = load_users()
        if "role" in users_df.columns:
            users_df["role"] = users_df["role"].str.lower()

        is_valid = False
        if role == "shift":
            is_valid = True
        else:
            match = users_df[
                (users_df["role"] == role) &
                (users_df["username"].str.casefold() == username.casefold()) &
                (users_df["password"] == password)
            ]
            is_valid = not match.empty

        if is_valid:
            # send user based on chosen role
            if role == "non_shift":
                return redirect(url_for("nonshift.index"))
            elif role == "shift":
                # keep your existing behavior or point elsewhere
                return redirect(url_for("main.index"))  # temp while testing
            elif role == "avp_vp":
                return redirect(url_for("nonshift.index"))  # temp while testing

            # fallback if role is unexpected
            return redirect(url_for("nonshift.index"))

        flash("❌ Invalid username or password.")

    # Show your existing home form template
    return render_template("home.html")

