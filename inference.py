#  MIT License
#   
#  Copyright (c) 2022 Adithya Singh, Rishabh Singh, Divyansh Agrawal
#   
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#   
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#   
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE. 


import json
import pickle
from flask import Flask, request

# initialize the app and load saved model and vocabulary
app = Flask(__name__)
model = pickle.load(open('RandomForestClassifier.sav', 'rb'))
vectorizer = pickle.load(open('Vectorizer.sav', 'rb'))

def preprocess(text):
    '''
    Function to preprocess the dataset by removing unwanted 
    spaces and characters
    '''
    # convert to lower case
    text = text.lower()
    # remove unwanted chracters
    text = text.replace("[^a-zA-Z]", " ")
    # remove unwanted whitespace
    text = text.replace("\s+", " ")
    return text

@app.route('/brand', methods=['POST'])
def inference():
    '''
    Serving function to process the HTTP requests
    '''
    # load the text body from curl request
    jsonText = json.loads(request.data)
    text = jsonText["text"]
    # preprocess the text
    text = preprocess(text)
    # obtain the feature embedding of given text
    text = vectorizer.transform([text])
    # make predictions and return the output
    pred = model.predict_proba(text)[0].tolist()
    prob = max(pred)
    label = model.classes_[pred.index(prob)]
    response = {'brand':label, 'probability':prob}
    return json.dumps([response])


if __name__=='__main__':
    # run the server
    app.run(host="0.0.0.0", port=5555, debug=True)