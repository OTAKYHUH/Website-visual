from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import threading
import webbrowser
import plotly.express as px
import openpyxl
from datetime import date  # at top
from routes.dashboard import dashboard
from routes.entry import entry
from routes.home import home
from routes.NonShift import nonshift

app = Flask(__name__)
app.secret_key = "vanguardoverdress"  # ← Required for flash and sessions

app.register_blueprint(dashboard)
app.register_blueprint(entry)
app.register_blueprint(home)
app.register_blueprint(nonshift)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Launch browser in a separate thread
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False)