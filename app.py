import uvicorn
from fastapi import FastAPI
from src.MamaFeature import MamaFeatures
import numpy as np
import pickle
import pandas as pd

app=FastAPI()
pickle_in = open('model.pkl','rb')
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return{'message':'Hello, doctor'}

@app.post('/predict')
def predict_diagnostic(data:MamaFeatures):
    data=data.model_dump()
    print(data)
    print('Hello')
    Radius_mean=data['Radius_mean']
    Texture_mean=data['Texture_mean']
    perimeter_mean=data['perimeter_mean']
    area_mean=data['area_mean']
    smoothness_mean=data['smoothness_mean']
    compactness_mean=data['compactness_mean']
    concavity_mean=data['concavity_mean']
    concave_points_mean=data['concave_points_mean']
    symmetry_mean=data['symmetry_mean']
    fractal_dimension_mean=data['fractal_dimension_mean']
    radius_se=data['radius_se']
    texture_se=data['texture_se']
    perimeter_se=data['perimeter_se']
    area_se=data['area_se']
    smoothness_se=data['smoothness_se']
    compactness_se=data['compactness_se']
    concavity_se=data['concavity_se']
    concave_points_se=data['concave_points_se']
    symmetry_se=data['symmetry_se']
    fractal_dimension_se=data['fractal_dimension_se']
    radius_wors=data['radius_wors']
    texture_worst=data['texture_worst']
    perimeter_worst=data['perimeter_worst']
    area_worst=data['area_worst']
    smoothness_worst=data['smoothness_worst']
    compactness_worst=data['compactness_worst']
    concavity_worst=data['concavity_worst']
    concave_points_worst=data['concave_points_worst']
    symmetry_worst=data['symmetry_worst']
    fractal_dimension_worst=data['fractal_dimension_worst']
    prediction = classifier.predict([[Radius_mean,
                                    Texture_mean,
                                    perimeter_mean,
                                    area_mean,
                                    smoothness_mean,
                                    compactness_mean,
                                    concavity_mean,        
                                    concave_points_mean,
                                    symmetry_mean,
                                    fractal_dimension_mean,
                                    radius_se,
                                    texture_se,
                                    perimeter_se,
                                    area_se,
                                    smoothness_se,
                                    compactness_se,
                                    concavity_se,
                                    concave_points_se,
                                    symmetry_se,
                                    fractal_dimension_se,
                                    radius_wors,
                                    texture_worst,
                                    perimeter_worst,
                                    area_worst,
                                    smoothness_worst,
                                    compactness_worst,
                                    concavity_worst,
                                    concave_points_worst,
                                    symmetry_worst,
                                    fractal_dimension_worst]])
    if(prediction=='B'):
        pred="Benign Tumor"
    else:
        pred="Malignant Tumor"
    return {
        'prediction': pred
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
