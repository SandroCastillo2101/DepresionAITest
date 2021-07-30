from sklearn import tree
from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import Flask,render_template,request,jsonify,redirect
from imblearn.over_sampling import SMOTE

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

    smt = SMOTE(random_state=42)
    X_smt, y_smt = smt.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def print_mc(matriz_conf):
    matriz_conf = pd.DataFrame(matriz_conf)
    matriz_conf.index = ["Real_0","Real_1"]
    matriz_conf.columns = ["Pred_0","Pred_1"]
    print(matriz_conf) 

def fx_evaluate_classif(y_real, pred, pred_proba):
    from sklearn import metrics as mt
    matriz_conf = mt.confusion_matrix(y_real,pred)
    print_mc(matriz_conf)
    roc = mt.roc_auc_score(y_real,pred_proba)
    accuracy_real = mt.accuracy_score(y_real,pred)
    print("\nROC: ", roc) 
    print("Accu:", accuracy_real,'\n')
    print(mt.classification_report(y_real, pred)[0:163])

def J48(X_train, X_test, y_train, y_test):

    model_rf = ensemble.RandomForestClassifier(n_estimators=157,min_samples_split=3,min_samples_leaf=1,max_features='auto',max_depth=8,bootstrap=False,
                                  n_jobs = 4,random_state=49)
    model_rf.fit(X_train, y_train)
    model_smt = model_rf
    
    # Generar las predicciones:
    y_pred_train_smt= model_smt.predict(X_train)
    y_pred_test_smt= model_smt.predict(X_test)

    # Generar las probabilidades
    y_pred_proba_train_smt= model_smt.predict_proba(X_train)[:,1]
    y_pred_proba_test_smt= model_smt.predict_proba(X_test)[:,1]

    print("Metricas del Training..." + "\n")
    fx_evaluate_classif(y_train, y_pred_train_smt, y_pred_proba_train_smt)
    print("Metricas del Testing..." + "\n")
    fx_evaluate_classif(y_test, y_pred_test_smt, y_pred_proba_test_smt)

    return model_smt

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