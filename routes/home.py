from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
from pathlib import Path

home = Blueprint('home', __name__)

USERS_FILE = (Path(__file__).resolve().parents[1] / "data" / "users.xlsx")

ROLE_TO_PBI_URL = {
    "shift":     "https://app.powerbi.com/reportEmbed?reportId=da709d26-ef86-4a23-bdfe-46ca2d260554&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3",
    "non_shift": "https://app.powerbi.com/reportEmbed?reportId=3a1ec64e-1b23-4811-a593-52ddb00ca52c&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3",
}

def load_users():
    if not USERS_FILE.exists():
        return pd.DataFrame(columns=["role", "username", "password"])
    df = pd.read_excel(USERS_FILE, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["role", "username", "password"]:
        df[col] = df[col].astype(str).str.strip()
    return df

@home.route("/", methods=["GET", "POST"])
def role_selection():
    if request.method == "POST":
        role = (request.form.get("role") or "").strip().lower()
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        users_df = load_users()
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
            pbi_url = ROLE_TO_PBI_URL.get(role)
            if pbi_url:
                # Just render the launch page, no redirect
                return render_template("launch_pbi.html", pbi_url=pbi_url)
            return redirect(url_for("home.role_selection"))

        flash("❌ Invalid username or password.")

    return render_template("home.html")
