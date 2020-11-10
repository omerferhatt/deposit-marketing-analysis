import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from models.net import get_mlp_model

# Disabling tensorflow info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def time_it(func):
	def function_wrapper(*args, **kwargs):
		tic = time.time()
		res = func(*args, **kwargs)
		toc = time.time() - tic
		print(f'Function `{func.__name__}` duration: {toc:.4f} seconds')
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
	# mm = MinMaxScaler()
	# scaled = mm.fit_transform(numerical_data)
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
	# Get categories without 'unknown' feature
	for i, unknown_col in enumerate(unknown_data.T):
		restored = missing_knn(true_array=np.concatenate([scaled, known_data, y], axis=1), missing_array=np.array(unknown_col))
		unknown_data[:, i] = restored
	# Replace all 'unknown' labels with zero binary equivalent
	ohe_unknown = OneHotEncoder(drop='first')
	unknown_data = ohe_unknown.fit_transform(unknown_data).toarray()
	# Creating pandas.DataFrame instance to save and have a quick review data
	new_df = pd.DataFrame(data=np.concatenate([scaled, known_data, unknown_data, y], axis=1))

	model_input_shape = (new_df.shape[-1] - 1,)
	num_classes = y.shape[-1]
	return [new_df, model_input_shape, num_classes]


def missing_knn(true_array: np.ndarray, missing_array: np.array) -> np.ndarray:
	"""
	Tries to fix unknown data features according to known features
	:param true_array: Feature known data
	:param missing_array: Data that some of features are missing
	:return: Fixed unknown data array
	"""
	# Create an copy of missing array, restored unknown labels will be change later
	restored = np.copy(missing_array)
	X_train, y_train = true_array[missing_array != arg.unknown_label, :], missing_array[missing_array != arg.unknown_label]
	X_restore = true_array[missing_array == arg.unknown_label, :]
	knn = KNeighborsClassifier(5)
	knn.fit(X_train, y_train)
	y_restore = knn.predict(X_restore)
	restored[restored == arg.unknown_label] = y_restore
	return restored


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


def run_mlp(X_train, y_train, X_test, y_test, input_shape, num_classes):
	"""
	Running MLP model instance, train and evaluate.

	:param X_train: Train features
	:param y_train: Train label
	:param X_test: Test features
	:param y_test: Test label
	:param input_shape: MLP input shape as a tuple
	:param num_classes: Total class number to construct output layer
	:return: Accuracy metric
	"""
	print('Running MLP!')
	# Crating MLP model and compiling
	model = get_mlp_model(input_shape, num_classes)
	# Fitting data into model
	model.fit(x=X_train, y=y_train, epochs=30, batch_size=32, verbose=0, workers=8)
	# Evaluation with test data
	_, acc = model.evaluate(x=X_test, y=y_test, batch_size=32, verbose=0, workers=8)
	print(f'\tMLP accuracy: {acc:.4f}')
	return acc


def run_knn(X_train, y_train, X_test, y_test):
	"""
	Running KNN model instance, train and evaluate.

	:param X_train: Train features
	:param y_train: Train label
	:param X_test: Test features
	:param y_test: Test label
	:return: Accuracy metric
	"""
	print('Running KNN')
	# Crating KNN model
	knn = KNeighborsClassifier(7)
	# Fitting data into model
	knn.fit(X_train, y_train.ravel())
	# Evaluation with test data
	acc = knn.score(X_test, y_test.ravel())
	print(f'\tKNN accuracy: {acc:.4f}')
	return acc


def run_r_forest(X_train, y_train, X_test, y_test):
	"""
	Running Random Forest model instance, train and evaluate.

	:param X_train: Train features
	:param y_train: Train label
	:param X_test: Test features
	:param y_test: Test label
	:return: Accuracy metric
	"""
	print('Running Random Forest')
	# Crating Random Forest model
	r_forest = RandomForestClassifier()
	# Fitting data into model
	r_forest.fit(X_train, y_train.ravel())
	# Evaluation with test data
	acc = r_forest.score(X_test, y_test.ravel())
	print(f'\tRandom Forest accuracy: {acc:.4f}')
	return acc


def main():
	# Loading *.csv file to runtime
	raw_dataframe = pd.read_csv(arg.csv_path)
	# Preprocessing
	preprocessed_dataframe, input_shape, num_classes = preprocess(df=raw_dataframe)
	# Pre-training for get X and y
	X, y = pre_training(df=preprocessed_dataframe)
	# Creating K-Fold
	fold = 5
	kf = KFold(n_splits=fold, shuffle=True, random_state=0)
	# Model metrics will be saved to history
	history = {'mlp': [], 'knn': [], 'r_forest': []}
	for n_fold, (train_index, test_index) in enumerate(kf.split(X)):
		print(f'\nK-Fold fold: {n_fold + 1}/{fold}')
		# Splitting data into folds
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print('Training started!')
		mlp_acc = run_mlp(X_train, y_train * 0.9, X_test, y_test, input_shape, num_classes)
		knn_acc = run_knn(X_train, y_train, X_test, y_test)
		r_forest_acc = run_r_forest(X_train, y_train, X_test, y_test)
		history['mlp'].append(mlp_acc)
		history['knn'].append(knn_acc)
		history['r_forest'].append(r_forest_acc)

	# Printing average results
	print('\n--------------------')
	print(f'Total fold count: {fold}')
	print(f"Average accuracy neural networks: {np.mean(np.array(history['mlp'])):.4f}")
	print(f"Average accuracy k-nearest neighbor: {np.mean(np.array(history['knn'])):.4f}")
	print(f"Average accuracy random forest: {np.mean(np.array(history['r_forest'])):.4f}")
	print('--------------------')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv-path', type=str, default='data/term-deposit-marketing-2020.csv')
	parser.add_argument('--unknown-label', type=str, default='unknown')
	arg = parser.parse_args()
	main()
