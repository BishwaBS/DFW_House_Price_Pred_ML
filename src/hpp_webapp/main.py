import numpy as np
import scipy
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd

app = Flask(__name__)

#loading scaler and model pickel file
scaler = pickle.load(open("minmax_scaler.pkl", 'rb'))
model = pickle.load(open('xgb_best_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


#For prediction, users will be required to pass in 5 feature value:
        # size
        # num of beds  
        # num of bathrooms  
        # distance_2_park 
        # distance_2_school
        # distance_2_hospital

#Please note that only ['size', 'dis_to_school', 'dis_to_hospital', 'dis_to_park', 'housequality'] is used in the backend 
# by the model. Housequality(hq) is calculated suing 'size', 'bed', 'bath'


## API to handle user request coming from form fill up in the web-app
@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    event = list(map(np.float64, features))
    hq = (event[0]- (event[1]*132) - (event[2]*40))
    featlist= [event[i] for i in [0,4,5, 3]] + [hq] ## it has to match the order of scaler
    arr = np.array(featlist).reshape(1,-1)
    final_features = scaler.transform(arr)
    prediction = model.predict(final_features)
    output = int(prediction[0])
    return render_template('index.html', prediction_text="House Price", prediction_val="${}".format(output))

## RESTful API to handle user request in json format
@app.route('/predictapi',methods=['POST'])
def predict_api():
    
    json_ = request.json
    trialdf = pd.DataFrame(json_)
    print(trialdf)
    hq = (trialdf.iloc[:,0]- (trialdf.iloc[:,1]*132) - (trialdf.iloc[:,2]*40))
    sel=trialdf.iloc[:, [0,4,5,3]]
    sel['housequality'] = hq
    features=scaler.transform(sel)
    output=model.predict(features)

    
    print(output)
    return jsonify({
               "Predicted house price":str(output)
           })

def main():
    cache.init_app(app, config=your_cache_config)

    with app.app_context():
        cache.clear()


if __name__ == "__main__":
    app.run(debug=True)
