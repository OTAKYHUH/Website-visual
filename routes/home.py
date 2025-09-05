from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
import os

home = Blueprint('home', __name__)

USERS_FILE = r"C:/Users/ahgua/Documents/Website visual/users.xlsx"

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["role", "username", "password"])
    return pd.read_excel(USERS_FILE)

@home.route("/", methods=["GET", "POST"])
def role_selection():
    if request.method == "POST":
        role = request.form.get("role")
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        print(f"🔍 POST - Role: {role}, Username: {username}, Password: {password}")

        # Shift Team – no login required
        if role == "shift":
            print("✅ Shift Team login")
            return redirect(url_for("dashboard.index"))

        # Load users from Excel
        users_df = load_users()

        match = users_df[
            (users_df["role"] == role) &
            (users_df["username"] == username) &
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
