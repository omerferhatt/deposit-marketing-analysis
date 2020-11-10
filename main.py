import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold


from models.net import get_mlp_model

CSV_PATH = 'data/term-deposit-marketing-2020.csv'
UNKNOWN_LABEL = 'unknown'


def time_it(func):
	def function_wrapper(*args, **kwargs):
		tic = time.time()
		res = func(*args, **kwargs)
		toc = time.time() - tic
		print(f'Function `{func.__name__}` duration: {toc} seconds')
		return res
	return function_wrapper


@time_it
def preprocess(df: pd.DataFrame) -> list:
	"""
	Preprocess the raw data and creates suitable one for machine learning.
	Standard scale on numerical features, encoding on categorical features. Dummy variables and
	unknown features handling.

	:param df: Raw dataframe instance
	:return: Preprocessed dataframe instance. Last column is prediction column.
	"""
	# Reading column names
	cols = df.columns.values

	# Data reorder according to feature space
	cols_numerical = ['age', 'balance', 'duration', 'day', 'campaign']
	cols_known = ['default', 'housing', 'loan', 'month', 'marital']

	# Order -> Numerical - Binary - Categorical(Known) - Categorical(Unknown) - Y
	new_columns = cols_numerical + cols_known + (df.columns.drop(cols_numerical + cols_known).tolist())
	df = df[new_columns]

	# Standardization on numerical features with StandardScaler
	numerical_data = df.iloc[:, :5].values
	ss = StandardScaler()
	scaled = ss.fit_transform(numerical_data)

	# Categorical feature encoding with OneHotEncoder,
	# dropping first feature removes dummy variable and helps to regularizing model
	known_data = df.iloc[:, 5:10].values
	ohe_known = OneHotEncoder(drop='first')
	known_data = ohe_known.fit_transform(known_data).toarray()

	# Creating different encoder for label column `y`
	y = df.iloc[:, -1].values.reshape(-1, 1)
	ohe_y = OneHotEncoder(drop='if_binary')
	y = ohe_y.fit_transform(y).toarray().reshape(-1, 1)

	# Unknown feature handling
	unknown_data = df.iloc[:, 10:-1].values
	unknown_categories = [list(set(feature)) for feature in unknown_data.T]
	# Get categories without 'unknown' feature
	[sub_category.remove(UNKNOWN_LABEL) for sub_category in unknown_categories]
	# Replace all 'unknown' labels with zero binary equivalent
	ohe_unknown = OneHotEncoder(handle_unknown='ignore', categories=unknown_categories)
	unknown_data = ohe_unknown.fit_transform(unknown_data).toarray()
	# Creating pandas.DataFrame instance to save and have a quick review data
	new_df = pd.DataFrame(data=np.concatenate([scaled, known_data, unknown_data, y], axis=1))

	model_input_shape = (new_df.shape[-1] - 1,)
	num_classes = y.shape[-1]
	return [new_df, model_input_shape, num_classes]


@time_it
def pre_training(df: pd.DataFrame) -> list:
	"""
	Splits dataset into X and y.

	:param df: Preprocessed dataframe instance
	:return: List with 2 array
	"""

	# Reading X and y from dataframe, reshaping y is necessary for shape assertion
	X = df.iloc[:, :-1].values
	y = df.iloc[:, -1].values.reshape(-1, 1)
	return [X, y]


if __name__ == '__main__':
	# Loading *.csv file to runtime
	raw_dataframe = pd.read_csv(CSV_PATH)
	# Preprocessing
	preprocessed_dataframe, input_shape, num_classes = preprocess(df=raw_dataframe)
	# Pre-training for get X and y
	X, y = pre_training(df=preprocessed_dataframe)
	# Creating K-Fold
	fold = 5
	kf = KFold(n_splits=fold, shuffle=True, random_state=0)
	# Crating MLP model and compiling
	model = get_mlp_model(input_shape, num_classes)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	# Model metrics will be saved to history
	history = []

	for n_fold, (train_index, test_index) in enumerate(kf.split(X)):
		print(f'K-Fold fold: {n_fold + 1}/{fold}')
		# Splitting data into folds
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# Fitting data into model
		print('Training started!')
		model.fit(x=X_train, y=y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0, workers=8)
		# Evaluation with test data and saving score into list
		loss, acc = model.evaluate(x=X_test, y=y_test, batch_size=16, verbose=2, workers=8)
		history.append([loss, acc])

	print('--------------------')
	print(f'Total fold count: {fold}')
	print(f'Average loss: {np.mean(np.array(history)[:, 0]):.3f}')
	print(f'Average accuracy: {np.mean(np.array(history)[:, 1]):.3f}')
	print('--------------------')