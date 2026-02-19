# routes/home.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os

home = Blueprint("home", __name__)

# ===================== PASSWORDS (SERVER-SIDE) =====================
# Set these in your server env vars (PythonAnywhere recommended).
STAFF_PROFILE_PASSWORD = os.environ.get("STAFF_PROFILE_PASSWORD", "cess")
DAILY_ANALYSIS_PASSWORD = os.environ.get("DAILY_ANALYSIS_PASSWORD", "cess")
SO_PERFORMANCE_PASSWORD = os.environ.get("SO_PERFORMANCE_PASSWORD", "cess")

# ===================== URLS (SERVER-SIDE) =====================

# ✅ Staff Profile URL (kept server-side)
STAFF_PROFILE_URL = (
    "https://globalpsa.sharepoint.com/:x:/s/PSAC-CNBD-YOD-efile/"
    "IQDQRbYZAnJeSaHbuUxC4x6yAXaNJ1Bwi693i6NtMbvJPZg?e=Tbd92g"
)

# ✅ Daily Analysis Power BI URL (kept server-side)
DAILY_ANALYSIS_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=8f105737-0465-44ee-822a-0791181fc5ca"
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

# ===================== HOME PAGE =====================

@home.route("/", methods=["GET", "POST"])
def role_selection():
    """
    This page is mostly driven by JS buttons.
    We keep POST for compatibility, but we no longer authenticate using users.xlsx.
    """
    if request.method == "POST":
        # If anything posts here accidentally, just go back with a message.
        flash("Please use the login popup for protected pages.")
        return redirect(url_for("home.role_selection"))

    return render_template("home.html")


# ===================== STAFF PROFILE (PASSWORD PROTECTED) =====================

@home.route("/staff-profile/login", methods=["POST"])
def staff_profile_login():
    pw = (request.form.get("password") or "").strip()

    if pw != STAFF_PROFILE_PASSWORD:
        flash("❌ Invalid password for Staff Profile.")
        return redirect(url_for("home.role_selection"))

    session["staff_profile_ok"] = True
    return redirect(url_for("home.staff_profile"))


@home.route("/staff-profile", methods=["GET"])
def staff_profile():
    if not session.get("staff_profile_ok"):
        flash("❌ Please enter Staff Profile password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(STAFF_PROFILE_URL)


# ===================== DAILY ANALYSIS (PASSWORD PROTECTED) =====================

@home.route("/daily-analysis/login", methods=["POST"])
def daily_analysis_login():
    pw = (request.form.get("password") or "").strip()

    if pw != DAILY_ANALYSIS_PASSWORD:
        flash("❌ Invalid password for Daily Analysis.")
        return redirect(url_for("home.role_selection"))

    session["daily_analysis_ok"] = True
    return redirect(url_for("home.daily_analysis"))


@home.route("/daily-analysis", methods=["GET"])
def daily_analysis():
    if not session.get("daily_analysis_ok"):
        flash("❌ Please enter Daily Analysis password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(DAILY_ANALYSIS_URL)


# ===================== SO PERFORMANCE (PASSWORD PROTECTED) =====================

@home.route("/so-performance/login", methods=["POST"])
def so_performance_login():
    pw = (request.form.get("password") or "").strip()

    if pw != SO_PERFORMANCE_PASSWORD:
        flash("❌ Invalid password for SO Individual Performance.")
        return redirect(url_for("home.role_selection"))

    session["so_performance_ok"] = True
    return redirect(url_for("home.so_performance"))


@home.route("/so-performance", methods=["GET"])
def so_performance():
    """
    Redirect to your existing SO Performance page.
    Previously: redirect(url_for("main.index"))
    """
    if not session.get("so_performance_ok"):
        flash("❌ Please enter SO Performance password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(url_for("main.index"))


# ===================== TRAINING PAGE (PUBLIC) =====================

@home.route("/training", methods=["GET"])
def training():
    return render_template("training.html")
