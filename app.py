from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import Flask,render_template,request,jsonify,redirect

def preprocess():
    input_file = "Depresion_v2.csv"
    df = pd.read_csv(input_file)

    df = df.replace("M", 0)
    df = df.replace("F", 1)
    df = df.replace("Soltero", 0)
    df = df.replace("Conviviente", 1)
    df = df.replace("Casado", 2)
    df = df.replace("Divorciado", 3)
    df = df.replace("Viudo", 4)
    df = df.replace("Si", 1)
    df = df.replace("No", 0)

    X = df.iloc[:, 1:-1]
    y = df["Depresion"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

def J48(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    return clf

def OneTest(model, x):
    x[0][0] = request.form['edad']
    x[0][1] = request.form['genero']
    x[0][2] = request.form['estado_civil']
    x[0][3] = request.form['preg_1']
    x[0][4] = request.form['preg_2']
    x[0][5] = request.form['preg_3']
    x[0][6] = request.form['preg_4']
    x[0][7] = request.form['preg_5']
    x[0][8] = request.form['preg_6']
    x[0][9] = request.form['preg_7']
    x[0][10] = request.form['preg_8']
    x[0][11] = request.form['preg_9']
    x[0][12] = request.form['preg_10']
    x[0][13] = request.form['preg_11']
    x[0][14] = request.form['preg_12']
    x[0][15] = request.form['preg_13']
    x[0][16] = request.form['preg_14']
    x[0][17] = request.form['preg_15']
    x[0][18] = request.form['preg_16']
    x[0][19] = request.form['preg_17']
    x[0][20] = request.form['preg_18']
    x[0][21] = request.form['preg_19']
    x[0][22] = request.form['preg_20']
    x[0][23] = request.form['preg_21']
    return model.predict(x)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def welcome():
	return render_template('index.html')

@app.route('/index.html', methods=['GET', 'POST'])
def bienvenido():
	return render_template('index.html')

@app.route('/test.html', methods=['GET','POST'])
def test():
    x = [[0 for i in range(24)]]
    if request.method=='POST':
        X_train, X_test, y_train, y_test = preprocess()
        model = J48(X_train, X_test, y_train, y_test)
        result = OneTest(model, x)
        if (result[0]==0):
            result = "Sin depresión"
        elif (result[0] == 1):
            result = "Con depresión"
        return render_template('result.html', resultado=result)
        
    elif request.method=='GET':
        return render_template('test.html')

@app.route('/result.html', methods=['GET', 'POST'])
def resultado():
	return render_template('result.html')

app.run(host='0.0.0.0', port=5000, debug=True)