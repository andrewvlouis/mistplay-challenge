from flask import Flask, request, jsonify
import re
import pickle

from flask import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
        
     # the querydf must be a dataframe in the same format as the csv input- i.e. x1....x26 
        
     # we make query_df into same input as trained model input
        
     query_df = query_df[['x2' , 'x4','x5','x6', 'x7','x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x26']] # take the same columns we trained the model on
     query_df['x2'] = query_df['x2'].astype(str) # make version a string
     query_df['x26'] = query_df['x26'].fillna('Unknown') # fill na with unknown
     query_df['x26'] = query_df['x26'].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x)) #r emove spaces and funky chars from string
     query = pd.get_dummies(query_df).as_matrix() # reshape into right format, xg ONLY accepts 2d arrays
    
     # make our prediction
     prediction = model.predict(query)
        
     # return our prediction
     return jsonify({'prediction': list(predictions)})
    
    
    
if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     app.run(port=5000, debug = True)

    
 