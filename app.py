import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_rf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    one = ['yes', 'present', 'good', 'normal', 'Yes', 'Present', 'Good', 'Normal', 'YES', 'PRESENT', 'GOOD', 'NORMAL']
    zero = ['no', 'notpresent', 'poor', 'abnormal', 'No', 'Notpresent', 'NotPresent', 'Poor', 'Abnormal', 'AbNormal', 'NO', 'NOTPRESENT', 'POOR', 'ABNORMAL']
    int_features = []
    for i in request.form.values():
        if i in one:
            int_features.append(1.0)
        elif i in zero:
            int_features.append(0.0)
        else:
            int_features.append(float(i))
            
    final_features = [np.array(int_features)]
    
    
    
    df = pd.read_csv("kidney.csv")
    df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']] = df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']].replace(to_replace={'\t8400':'8400', '\t6200':'6200', '\t43':'43', '\t?':np.nan, '\tyes':'yes', '\tno':'no', 'ckd\t':'ckd'})
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df[['pcv', 'wc', 'rc']] = df[['pcv', 'wc', 'rc']].astype('float64')
    df.drop('id',axis=1,inplace=True)
    col = ['rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']
    encoder = LabelEncoder()
    for col in col:
        df[col] = encoder.fit_transform(df[col])
    df[['appet', 'classification']] = df[['appet', 'classification']].replace(to_replace={'good':'1', 'ckd':'1', 'notckd':'0', 'poor':'0'})
    df[['classification', 'appet']] = df[['classification', 'appet']].astype('int64')
    X = df.drop("classification", axis=1)
    y = df["classification"]
    scaler = MinMaxScaler()
    features = scaler.fit_transform(X)

    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    
    output = prediction
    
    if output == [0]:
        output = "Kidney Disease Not Detected"
    elif output == [1]:
        output = "Kidney Disease Detected"
    
    return render_template('index.html', prediction_text='Diagnosis Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)