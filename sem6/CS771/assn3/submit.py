import numpy as np
import pandas as pd
import pickle as pkl

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
	
	# Load your model file

	with open( "model_o3", "rb" ) as file:
		reg_o3 = pkl.load( file )
	
	with open( "model_no2", "rb" ) as file:
		reg_no2 = pkl.load( file )
	
	# Make two sets of predictions, one for O3 and another for NO2

	df[ "bias_coef" ] = 1
	df = df.drop(["Time", "temp", "humidity"], axis=1)

	np_df = np.array(df)

	# np_df_o3 = np_df[:, 0]
	# np_df_no2 = np_df[:, 1]

	np_df_params = np_df[:, 0:5]
	
	pred_o3 = reg_o3.predict( np_df_params )
	pred_no2 = reg_no2.predict( np_df_params )
	
	# Return both sets of predictions
	return ( pred_o3, pred_no2 )
