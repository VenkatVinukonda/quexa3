import numpy as np
import pandas as pd
from flask import Flask, render_template, request,send_from_directory
import pickle
import datetime as dt
import calendar
import os

app = Flask(__name__)
loaded_model = pickle.load(open('final_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/another_page1')
def another_page1():
    return render_template('Holiday.html') 
@app.route('/another_page2')
def another_page2():
    return render_template('Comparitive.html') 
@app.route('/another_page3')
def another_page3():
    return render_template('Trend.html') 
@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory('static', filename)
@app.route('/pred', methods=['POST'])
def predict():
    store = request.form.get('store')
    size = request.form.get('size')
    department = request.form.get('department')
    temperature = request.form.get('temperature')
    date = request.form.get('date')
    isHoliday = request.form.get('isHoliday') 
    isHoliday=int(isHoliday)

    # Check if date is provided and not empty
    if date:
        # Parse the date string to datetime
        d = dt.datetime.strptime(date, '%Y-%m-%d')
        year = d.year
        month = d.month
        month_name = calendar.month_name[month]
    else:
        # Provide default values if date is not provided
        year = 2023
        month = 1
        month_name = calendar.month_name[month] 
    X_test =[[
        store,
        department,
        size, 
        isHoliday, 
        212,
        temperature,
        0,
        1, 
        2050,
        month,
        year]]
    
    ypred=loaded_model.predict(X_test)
    output = round(ypred[0], 2)

    return render_template('index.html',store=store,department=department,month=month,year=year,output=output)

port = os.getenv('VCAP_APP_PORT', '8080')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=False, host='0.0.0.0', port=port)
