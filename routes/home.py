# home.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from pathlib import Path
import pandas as pd

home = Blueprint("home", __name__)

# Adjust this to your actual users.xlsx location if different
USERS_FILE = (Path(__file__).resolve().parents[1] / "data" / "users.xlsx")

# ✅ Staff Profile password (server-side)
STAFF_PROFILE_PASSWORD = "cess"

# ✅ Staff Profile URL (kept server-side)
STAFF_PROFILE_URL = (
    "https://globalpsa.sharepoint.com/:x:/s/PSAC-CNBD-YOD-efile/"
    "IQDQRbYZAnJeSaHbuUxC4x6yAXaNJ1Bwi693i6NtMbvJPZg?e=Tbd92g"
)

def load_users():
    if not USERS_FILE.exists():
        # Safe default schema
        return pd.DataFrame(columns=["role", "username", "password"])
    df = pd.read_excel(USERS_FILE, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["role", "username", "password"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(str).str.strip()
    return df


@home.route("/", methods=["GET", "POST"])
def role_selection():
    """
    GET  -> show the landing page (buttons for Shift / Non-Shift / AVP/VP)
    POST -> handle Non-Shift and AVP/VP only (Shift opens Power BI via JS on the client)
    """
    if request.method == "POST":
        role = (request.form.get("role") or "").strip().lower()
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        users_df = load_users()
        if "role" in users_df.columns:
            users_df["role"] = users_df["role"].str.lower()

        # Shift is handled on the client; ignore here
        if role == "non_shift":
            match = users_df[
                (users_df["role"] == role) &
                (users_df["username"].str.casefold() == username.casefold()) &
                (users_df["password"] == password)
            ]
            if not match.empty:
                return redirect(url_for("main.index"))
            flash("❌ Invalid username or password for Non-Shift.")

        elif role == "avp_vp":
            match = users_df[
                (users_df["role"] == role) &
                (users_df["username"].str.casefold() == username.casefold()) &
                (users_df["password"] == password)
            ]
            if not match.empty:
                return redirect(url_for("nonshift.index"))
            flash("❌ Invalid username or password for AVP/VP.")

        # Fallthrough: show page again with message (if any)
        return render_template("home.html")

    # GET
    return render_template("home.html")


# ===================== STAFF PROFILE (PASSWORD PROTECTED) =====================

@home.route("/staff-profile/login", methods=["POST"])
def staff_profile_login():
    """
    Handles Staff Profile password entry from the shared modal.
    Expects form field: password
    """
    pw = (request.form.get("password") or "").strip()

    if pw != STAFF_PROFILE_PASSWORD:
        flash("❌ Invalid password for Staff Profile.")
        return redirect(url_for("home.role_selection"))

    # Mark session as allowed
    session["staff_profile_ok"] = True
    return redirect(url_for("home.staff_profile"))


@home.route("/staff-profile", methods=["GET"])
def staff_profile():
    """
    Only accessible after successful staff_profile_login.
    Redirects to the SharePoint Staff Profile link.
    """
    if not session.get("staff_profile_ok"):
        flash("❌ Please enter Staff Profile password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(STAFF_PROFILE_URL)


@home.route("/training", methods=["GET"])
def training():
    """Public Training page (no password)."""
    return render_template("training.html")
