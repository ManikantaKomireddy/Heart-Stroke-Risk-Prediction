from distutils.log import debug
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3

app = Flask(__name__)

df = pd.read_csv('processed.csv')
del df['Unnamed: 0']

from sklearn import preprocessing
  
# label_encoder object knows 
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['gender']= label_encoder.fit_transform(df['gender'])
df['ever_married']= label_encoder.fit_transform(df['ever_married'])
df['work_type']= label_encoder.fit_transform(df['work_type'])
df['Residence_type']= label_encoder.fit_transform(df['Residence_type'])
df['smoking_status']= label_encoder.fit_transform(df['smoking_status'])

X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']]
y = df['stroke']


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  VotingClassifier
from sklearn.metrics import confusion_matrix
clf1 = RandomForestClassifier()
#clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = DecisionTreeClassifier()
eclf1 = VotingClassifier(estimators=[('nb', clf1),  ('dt', clf3)], voting='soft')
eclf1.fit(X, y)
predictions = eclf1.predict(X)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    
    
    model = joblib.load('model.sav')
    int_features= [float(x) for x in request.form.values()]
    final4=[np.array(int_features)]

    #predict = model.predict(final4)
    predict1 = eclf1.predict_proba(final4)
    print(predict1)
    #print(predict)
    predict = round(predict1[0][1])
    print(predict)
    #predict = round(predict[0], 2)
    #val = round(predict * 82.71,2)
    
    return render_template('result.html', result = predict)
    


@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')

@app.route('/about')
def about():
	return render_template('about.html')



if __name__ == "__main__":
    app.run(debug=True)
