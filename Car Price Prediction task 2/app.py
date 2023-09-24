from flask import Flask,render_template,request
from flask_material import Material
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

features=['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower',
		'citympg','highwaympg','fueltype_diesel','fueltype_gas','aspiration_std','aspiration_turbo',
		'doornumber_four','doornumber_two','carbody_convertible','carbody_hardtop','carbody_hatchback',
		'carbody_sedan','carbody_wagon','drivewheel_4wd','drivewheel_fwd','drivewheel_rwd','enginetype_dohc',
		'enginetype_dohcv','enginetype_l','enginetype_ohc','enginetype_ohcf','enginetype_ohcv','enginetype_rotor',
		'cylindernumber_eight','cylindernumber_five','cylindernumber_four','cylindernumber_six',
		'cylindernumber_three','cylindernumber_twelve','cylindernumber_two','fuelsystem_1bbl','fuelsystem_2bbl',
		'fuelsystem_4bbl','fuelsystem_idi','fuelsystem_mfi','fuelsystem_mpfi','fuelsystem_spdi','fuelsystem_spfi',
		'CarsRange_Budget','CarsRange_Medium','CarsRange_Highend']

nominal=['fueltype','aspiration','doornumber','carbody','drivewheel',
		'enginetype','cylindernumber','fuelsystem','CarsRange']

numcols=['wheelbase','carlength','carwidth','curbweight','enginesize',
		'boreratio','horsepower','citympg','highwaympg']

models = ["LinearRegression","DecisionTree","RandomForest","LGBM","XGBoost","CatBoost"]

df=pd.read_csv('CarPrice_Assignment.csv')
scaler = StandardScaler().fit(df[numcols])

features_dict = dict()

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		model_choice = request.form['model_choice']

		for f in features:
			features_dict[f] = False

		for f in numcols:
			num = float(request.form[f])
			features_dict[f] = num

		for f in nominal:
			s = f+'_'+request.form[f]
			features_dict[s] = True

		new_df=pd.DataFrame.from_dict([features_dict])

		new_df[numcols] = scaler.transform(new_df[numcols])
		new_df=new_df[features]

		# Reloading the Model
		if model_choice == 'lreg':
			model = pickle.load(open('LinearRegression.pkl', 'rb'))
			prediction = model.predict(new_df)
		elif model_choice == 'dtree':
			model = pickle.load(open('DecisionTree.pkl', 'rb'))
			prediction = model.predict(new_df)
		elif model_choice == 'rforest':
			model = pickle.load(open('RandomForest.pkl', 'rb'))
			prediction = model.predict(new_df)
		elif model_choice == 'XGB':
			model = pickle.load(open('XGBoost.pkl', 'rb'))
			prediction = model.predict(new_df)
		elif model_choice == 'LGBM':
			model = pickle.load(open('LGBM.pkl', 'rb'))
			prediction = model.predict(new_df)
		elif model_choice == 'CB':
			model = pickle.load(open('CatBoost.pkl', 'rb'))
			prediction = model.predict(new_df)
		elif model_choice == 'avg':
			average = 0.0
			for m in models:
				model = pickle.load(open(m+'.pkl', 'rb'))
				average += model.predict(new_df)/6

			prediction = average
		
		param = request.form


	return render_template('index.html',
		result_prediction=round(prediction[0],2),
		param = param,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(host="0.0.0.0",port=500,debug=True)