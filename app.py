from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

##route for homepage

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect form data and ensure proper naming and conversion
        data = CustomData(
            GRE_Score=int(request.form.get('GRE_Score')),
            TOEFL_Score=float(request.form.get('TOEFL_Score')),
            University_Rating=int(request.form.get('University_Rating')),
            SOP=float(request.form.get('SOP')),
            LOR=float(request.form.get('LOR')),
            CGPA=float(request.form.get('CGPA')),
            Research=int(request.form.get('Research'))
        )
        
        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        

        # Render results on the page
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

