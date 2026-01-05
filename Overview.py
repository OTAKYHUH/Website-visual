from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import threading
import webbrowser
import plotly.express as px
import openpyxl
from datetime import date  # at top
from routes.dashboard import dashboard
from routes.home import home
from routes.NonShift import nonshift
from routes.main import main
from routes.profile import profile
from routes.daily import daily
from routes.yas_booklet import yas_booklet_bp

app = Flask(__name__)
app.secret_key = "vanguardoverdress"  # ← Required for flash and sessions

app.register_blueprint(dashboard)
app.register_blueprint(home)
app.register_blueprint(nonshift)
app.register_blueprint(main)
app.register_blueprint(profile)
app.register_blueprint(daily)
app.register_blueprint(yas_booklet_bp)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Launch browser in a separate thread
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)