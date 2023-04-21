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



import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pickle


def preprocess(data):
    '''
    Function to preprocess the dataset by removing unwanted 
    spaces and characters
    '''
    # convert all characters to lower case
    data["raw_text"] = data["raw_text"].apply(lambda x: x.lower())

    # remove all non alphabetic characters
    data["raw_text"] = data["raw_text"].str.replace("[^a-zA-Z]", " ")

    # remove unnecessary whitespaces
    data["raw_text"] = data["raw_text"].str.replace("\s+", " ")
    return data

def featureExtraction(data):
    '''
    Function to convert text into feature embedding which can be
    learned by a supervised learning algorithm 
    '''
    # split the dataset into training and validation
    xTrain, xTest, yTrain, yTest = train_test_split(data["raw_text"], 
                                                    data["brand"], 
                                                    test_size=0.2, 
                                                    random_state=21)
    # initialize bag of words learning approach
    vectorizer = CountVectorizer(max_features=3000)

    # learn the vocabulary of training set and 
    # map the text features to trainable numeric features
    xTrain = vectorizer.fit_transform(xTrain)

    # map the text features of test dataset according 
    # to vocabulary learned in training set
    xTest = vectorizer.transform(xTest)

    # save the learned vocabulary mapping to be used in serving
    pickle.dump(vectorizer, open('Vectorizer.sav', 'wb+'))

    # oversample the data to address class imbalance problem
    ros = RandomOverSampler(random_state=0)
    xTrain, yTrain = ros.fit_resample(xTrain, yTrain)
    return xTrain, yTrain, xTest, yTest

def train(xTrain, yTrain):
    '''
    Function to train the random forest classification model
    '''
    # train random forest classifier
    model = RandomForestClassifier(n_estimators=1000, random_state=0)
    model.fit(xTrain, yTrain)
    return model

def evaluate(model, xTest, yTest):
    '''
    Function to evaluate the trained model's performance
    '''
    # get predictions on the test set
    yPred = model.predict(xTest)
    # compute precision, recall, f1 score and confusion matrix for test set
    classificationScore = classification_report(yTest, yPred)
    confusionMatrix = confusion_matrix(yTest, yPred)
    print("[INFO]: Output results on the test dataset")
    print("F1 Accuracy:\n", classificationScore)
    print("Confusion Matrix:\n", confusionMatrix)


if __name__=='__main__':
    print("[INFO]: Loading the dataset")
    with open('exercise_data.json', 'r') as f:
        json_data = json.load(f)

    data = pd.json_normalize(json_data)
    print("[INFO]: Preprocessing the dataset")
    preprocessedData = preprocess(data)
    print("[INFO]: Extracting trainable features")
    xTrain, yTrain, xTest, yTest = featureExtraction(preprocessedData)
    print("[INFO]: Running the training job")
    model = train(xTrain, yTrain)
    print("[INFO]: Evaluate the trained model")
    evaluate(model, xTest, yTest)
    print("[INFO]: Save the trained model")
    pickle.dump(model, open('RandomForestClassifier.sav', 'wb+'))
