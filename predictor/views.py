from django.shortcuts import render
import joblib
import os
import numpy as np
from django.conf import settings

# 1. Load the Model (We do this outside the function so it only loads once)
# We use os.path.join to make sure it works on Windows, Mac, and Linux
model_path = os.path.join(settings.BASE_DIR, 'predictor/models/heart_disease_model.pkl')
model = joblib.load(model_path)

def home(request):
    result = None
    
    if request.method == 'POST':
        # 2. Get data from the HTML form
        # We use 'float()' to convert text input to numbers
        try:
            age = float(request.POST['age'])
            sex = float(request.POST['sex'])
            cp = float(request.POST['cp'])
            trestbps = float(request.POST['trestbps'])
            chol = float(request.POST['chol'])
            fbs = float(request.POST['fbs'])
            restecg = float(request.POST['restecg'])
            thalach = float(request.POST['thalach'])
            exang = float(request.POST['exang'])
            oldpeak = float(request.POST['oldpeak'])
            slope = float(request.POST['slope'])
            ca = float(request.POST['ca'])
            thal = float(request.POST['thal'])

            # 3. Organize the data (Must match the order you trained on!)
            user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # 4. Make Prediction
            prediction = model.predict(user_data)
            
            # 5. Interpret Result
            if prediction[0] == 1:
                result = "High Risk of Heart Disease ⚠️"
            else:
                result = "Healthy Heart ✅"
                
        except Exception as e:
            result = f"Error: {e}"

    return render(request, 'home.html', {'result': result})