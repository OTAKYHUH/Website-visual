# routes/home.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os

home = Blueprint("home", __name__)

# ===================== PASSWORDS (SERVER-SIDE) =====================
# Set these in your server env vars (PythonAnywhere recommended).
SO_PERFORMANCE_1STHALF_PASSWORD = os.environ.get("SO_PERFORMANCE_1STHALF_PASSWORD", "cess20")
SO_PERFORMANCE_2NDHALF_PASSWORD = os.environ.get("SO_PERFORMANCE_2NDHALF_PASSWORD", "cess20")
JO_PERFORMANCE_PASSWORD = os.environ.get("JO_PERFORMANCE_PASSWORD", "cess20")
STAFF_PROFILE_PASSWORD = os.environ.get("STAFF_PROFILE_PASSWORD", "cess27")
PLAYBOOK_PASSWORD = os.environ.get("PLAYBOOK.PASSWORD", "cess27")
DAILY_ANALYSIS_PASSWORD = os.environ.get("DAILY_ANALYSIS_PASSWORD", "cess27")
PPT_WEEKLY_PASSWORD = os.environ.get("PPT_WEEKLY_PASSWORD", "cess27")
PPT_WEEKLY_SHIOK_PASSWORD = os.environ.get("PPT_WEEKLY_SHIOK_PASSWORD", "cess27")
SHIOK_PASSWORD = os.environ.get("SHIOK_PASSWORD", "cess27")
DIVERSION_PASSWORD = os.environ.get("DIVERSION_PASSWORD", "cess27")
ARMG_YC_PASSWORD = os.environ.get("ARMG_YC_PASSWORD", "cess27")

# ===================== URLS (SERVER-SIDE) =====================

SO_PERFORMANCE_1STHALF_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=6089e1a7-b291-4e3e-996c-ba9e3da9b503"
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

SO_PERFORMANCE_2NDHALF_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=5e35ecda-ca26-4ac8-ad92-50be568d2064"
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

JO_PERFORMANCE_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=f2de832e-d53a-4786-9a6a-24493ce5e91a"
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

STAFF_PROFILE_URL = (
    "https://globalpsa.sharepoint.com/:x:/s/PSAC-CNBD-YOD-efile/"
    "IQDQRbYZAnJeSaHbuUxC4x6yAXaNJ1Bwi693i6NtMbvJPZg?e=Tbd92g"
)

PLAYBOOK_URL = (
    "https://miro.com/app/dashboard/"
)

DAILY_ANALYSIS_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=8f105737-0465-44ee-822a-0791181fc5ca"
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

PPT_WEEKLY_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=08e074d4-3c03-46b9-b533-812c20defe3e" 
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

PPT_WEEKLY_SHIOK_URL = (
    "https://app.powerbi.com/reportEmbed?reportId=c250baad-0d3f-4420-b764-ff3dd9cd1510"
    "&autoAuth=true&ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3"
)

SHIOK_URL = (
    "https://app.powerbi.com/groups/me/reports/5b1088de-e6e3-4391-87bb-1af4756165b2/ReportSection0b72e683b08137da9f00?experience=power-bi&bookmarkGuid=Bookmarkdd9bb3a90dba0c15bd6a"
    )

DIVERSION_URL = (
    "https://app.powerbi.com/groups/me/apps/6418f333-7812-43d1-9e08-8def7120daed/reports/493c1ec9-1f54-41bf-a57b-79815922a571/ReportSection?ctid=bc1b92b9-5dc9-49be-995b-c97eb515a1d3&experience=power-bi"
    )

ARMG_YC_URL = (
    "https://app.powerbi.com/groups/me/reports/56b1a68c-0fbf-4c30-a1e1-a3f7c0f68635/ReportSection6f2ee9464148704944ec?experience=power-bi"
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

# ===================== PLAYBOOK (PASSWORD PROTECTED) =====================

@home.route("/playbook/login", methods=["POST"])
def playbook_login():
    pw = (request.form.get("password") or "").strip()

    if pw != PLAYBOOK_PASSWORD:
        flash("❌ Invalid password for Playbook.")
        return redirect(url_for("home.role_selection"))

    session["playbook_ok"] = True
    return redirect(url_for("home.playbook"))


@home.route("/playbook", methods=["GET"])
def playbook():
    
    if not session.get("playbook_ok"):
        flash("❌ Please enter Playbook password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(PLAYBOOK_URL)
# ===================== SO PERFORMANCE (Jan-Jun) (PASSWORD PROTECTED) =====================

@home.route("/so-performance-1sthalf/login", methods=["POST"])
def so_performance_1sthalf_login():
    pw = (request.form.get("password") or "").strip()

    if pw != SO_PERFORMANCE_1STHALF_PASSWORD:
        flash("❌ Invalid password for SO Individual Performance (January - June).")
        return redirect(url_for("home.role_selection"))

    session["so_performance_1sthalf_ok"] = True
    return redirect(url_for("home.so_performance_1sthalf"))


@home.route("/so-performance-1sthalf", methods=["GET"])
def so_performance_1sthalf():
    
    if not session.get("so_performance_1sthalf_ok"):
        flash("❌ Please enter SO Performance (January - June) password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(SO_PERFORMANCE_1STHALF_URL)

# ===================== SO PERFORMANCE (Jul-Dec) (PASSWORD PROTECTED) =====================

@home.route("/so-performance-2ndhalf/login", methods=["POST"])
def so_performance_2ndhalf_login():
    pw = (request.form.get("password") or "").strip()

    if pw != SO_PERFORMANCE_2NDHALF_PASSWORD:
        flash("❌ Invalid password for SO Individual Performance (July - December).")
        return redirect(url_for("home.role_selection"))

    session["so_performance_2ndhalf_ok"] = True
    return redirect(url_for("home.so_performance_2ndhalf"))


@home.route("/so-performance-2ndhalf", methods=["GET"])
def so_performance_2ndhalf():
    
    if not session.get("so_performance_2ndhalf_ok"):
        flash("❌ Please enter SO Performance (July to December) password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(SO_PERFORMANCE_2NDHALF_URL)


# ===================== JO PERFORMANCE (PASSWORD PROTECTED) =====================

@home.route("/jo-performance/login", methods=["POST"])
def jo_performance_login():
    pw = (request.form.get("password") or "").strip()

    if pw != JO_PERFORMANCE_PASSWORD:
        flash("❌ Invalid password for JO Individual Performance.")
        return redirect(url_for("home.role_selection"))

    session["jo_performance_ok"] = True
    return redirect(url_for("home.jo_performance"))


@home.route("/jo-performance", methods=["GET"])
def jo_performance():
    
    if not session.get("jo_performance_ok"):
        flash("❌ Please enter JO Performance password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(JO_PERFORMANCE_URL)

# ===================== PPT WEEKLY (PASSWORD PROTECTED) =====================

@home.route("/ppt-weekly/login", methods=["POST"])
def ppt_weekly_login():
    pw = (request.form.get("password") or "").strip()

    if pw != PPT_WEEKLY_PASSWORD:
        flash("❌ Invalid password for PPT Weekly.")
        return redirect(url_for("home.role_selection"))

    session["ppt_weekly_ok"] = True
    return redirect(url_for("home.ppt_weekly"))


@home.route("/ppt_weekly", methods=["GET"])
def ppt_weekly():
    
    if not session.get("ppt_weekly_ok"):
        flash("❌ Please enter PPT Weekly password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(PPT_WEEKLY_URL)

# ===================== PPT WEEKLY (SHIOK) (PASSWORD PROTECTED) =====================
@home.route("/ppt_weekly_shiok/login", methods=["POST"])
def ppt_weekly_shiok_login():
    pw = (request.form.get("password") or "").strip()

    if pw != PPT_WEEKLY_SHIOK_PASSWORD:
        flash("❌ Invalid password for PPT WEEKLY (SHIOK).")
        return redirect(url_for("home.role_selection"))

    session["ppt_weekly_shiok_ok"] = True
    return redirect(url_for("home.ppt_weekly_shiok"))


@home.route("/ppt_weekly_shiok", methods=["GET"])
def ppt_weekly_shiok():
    
    if not session.get("ppt_weekly_shiok_ok"):
        flash("❌ Please enter PPT WEEKLY (SHIOK) password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(PPT_WEEKLY_SHIOK_URL)

# ===================== SHIOK 1.1 (PASSWORD PROTECTED) =====================
@home.route("/shiok/login", methods=["POST"])
def shiok_login():
    pw = (request.form.get("password") or "").strip()

    if pw != SHIOK_PASSWORD:
        flash("❌ Invalid password for SHIOK 1.1.")
        return redirect(url_for("home.role_selection"))

    session["shiok_ok"] = True
    return redirect(url_for("home.shiok"))


@home.route("/shiok", methods=["GET"])
def shiok():
    
    if not session.get("shiok_ok"):
        flash("❌ Please enter SHIOK 1.1 password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(SHIOK_URL)

# ===================== DIVERSION (PASSWORD PROTECTED) =====================
@home.route("/diversion/login", methods=["POST"])
def diversion_login():
    pw = (request.form.get("password") or "").strip()

    if pw != DIVERSION_PASSWORD:
        flash("❌ Invalid password for Diversion.")
        return redirect(url_for("home.role_selection"))

    session["diversion_ok"] = True
    return redirect(url_for("home.diversion"))


@home.route("/diversion", methods=["GET"])
def diversion():
    
    if not session.get("diversion_ok"):
        flash("❌ Please enter Diversion password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(DIVERSION_URL)

# ===================== ARMG/YC (PASSWORD PROTECTED) =====================
@home.route("/armg_yc/login", methods=["POST"])
def armg_yc_login():
    pw = (request.form.get("password") or "").strip()

    if pw != ARMG_YC_PASSWORD:
        flash("❌ Invalid password for ARMG/YC.")
        return redirect(url_for("home.role_selection"))

    session["armg_yc_ok"] = True
    return redirect(url_for("home.armg_yc"))


@home.route("/armg_yc", methods=["GET"])
def armg_yc():
    
    if not session.get("armg_yc_ok"):
        flash("❌ Please enter ARMG/YC password first.")
        return redirect(url_for("home.role_selection"))

    return redirect(ARMG_YC_URL)
# ===================== TRAINING PAGE (PUBLIC) =====================

@home.route("/training", methods=["GET"])
def training():
    return render_template("training.html")
