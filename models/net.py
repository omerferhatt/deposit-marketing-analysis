from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L1, L2


def get_mlp_model(input_shape, num_classes):
	"""
	Creates MLP (Multi Layer Perceptron) model with specified input shape and output classes.

	:param input_shape: Input shape with out batch size.
	:param num_classes: Total class number to set-up output dense layer
	:return: tensorflow.keras.Sequential model instance
	"""
	model = Sequential(name='mlp_model')
	model.add(Dense(512, input_shape=input_shape, activation='relu', kernel_regularizer=L1(l1=0.005), name='input_dense_1'))
	model.add(Dense(256, activation='relu', kernel_regularizer=L2(l2=0.005), name='dense_2'))
	model.add(Dense(128, activation='relu', kernel_regularizer=L2(l2=0.005), name='dense_3'))
	model.add(Dense(64, activation='relu', kernel_regularizer=L2(l2=0.005), name='dense_4'))
	model.add(Dense(32, activation='relu', kernel_regularizer=L2(l2=0.005), name='dense_5'))
	model.add(Dense(num_classes, activation='sigmoid', name='output_dense_6'))
	return model
