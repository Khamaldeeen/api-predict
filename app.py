import numpy as np 
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import catboost



#load model
app = Flask(__name__)
api = Api(app)

model = joblib.load("myModel.sav")

class MakePrediction(Resource):
    @staticmethod
    def post():
        data = request.form.to_dict()
        prd = data["product"]
        cal = data["calories"]
        carb = data["carbs"]
        time = data["time"]
        dsh = data["dish"]
        heat = data["heat"]
        fat = data["fat"]
        ingrd = data["no_ingredients"]
        prot = data["proteins"]
        pro_clss = data["protein_class"]
        cuisine = data["cuisine"]
        answer = model.predict([prd, cal, carb, time, dsh, heat, fat, ingrd, prot, pro_clss, cuisine])
        answer = np.exp(answer)

        return jsonify({
            'Prediction': answer
        })
    
    
api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
