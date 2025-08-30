# app.py

from flask import Flask, request, render_template
import pandas as pd
import predict
import config

app = Flask(__name__)

# dropdowns options
job_options = ['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown']
marital_options = ['divorced','married','single','unknown']
education_options = ['basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown']
default_options = ['no','yes','unknown']
housing_options = ['no','yes','unknown']
loan_options = ['no','yes','unknown']
contact_options = ['cellular','telephone']
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week_options = ['mon','tue','wed','thu','fri']
poutcome_options = ['failure','nonexistent','success']


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    if request.method == 'POST':
        #Get data from the form
        form_data = request.form.to_dict()
        
        #Convert data types for numeric fields
        for col in config.NUM_COLS:
            if col in form_data:
                # Use float for fields that can have decimals, int for others
                form_data[col] = float(form_data[col]) if '.' in form_data[col] else int(form_data[col])

        #Make prediction
        prediction_result = predict.make_prediction(form_data)

    return render_template(
        'index.html', 
        prediction=prediction_result,
        # Pass options for dropdowns to the template
        job_options=job_options,
        marital_options=marital_options,
        education_options=education_options,
        default_options=default_options,
        housing_options=housing_options,
        loan_options=loan_options,
        contact_options=contact_options,
        month_options=month_options,
        day_of_week_options=day_of_week_options,
        poutcome_options=poutcome_options
    )


if __name__ == '__main__':
    app.run(debug=True)