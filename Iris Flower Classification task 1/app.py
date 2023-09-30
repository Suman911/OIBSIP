from flask import Flask,render_template,url_for,request
from flask_material import Material
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Initializing scaler and label_encoder
df=pd.read_csv('Iris.csv')
label_encoder = LabelEncoder().fit(df['Species'])
scaler = StandardScaler().fit(df.drop(['Id', 'Species'], axis=1).to_numpy())
df.drop(columns='Id', inplace=True)

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex = np.array(clean_data).reshape(1,-1)
		ex1 = scaler.transform(ex) 


		# Reloading the Model
		if model_choice == 'dense':
			logit_model = pickle.load(open('dense_model.pkl', 'rb'))
			result_prediction = logit_model.predict(ex1)
			prediction = np.argmax(result_prediction)
			result_prediction = label_encoder.inverse_transform([prediction])[0]
		elif model_choice == 'logistic':
			logit_model = pickle.load(open('logistic_regression.pkl', 'rb'))
			result_prediction = logit_model.predict(ex1)
			result_prediction = label_encoder.inverse_transform(result_prediction)[0]
		elif model_choice == 'knn':
			logit_model = pickle.load(open('KNN.pkl', 'rb'))
			result_prediction = logit_model.predict(ex1)
			result_prediction = label_encoder.inverse_transform(result_prediction)[0]
		elif model_choice == 'dtree':
			logit_model = pickle.load(open('decision_tree.pkl', 'rb'))
			result_prediction = logit_model.predict(ex1)
			result_prediction = label_encoder.inverse_transform(result_prediction)[0]

	return render_template('index.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(host="0.0.0.0",port=500,debug=True)