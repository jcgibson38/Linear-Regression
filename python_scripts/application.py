import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.py import least_squares_fit

# Create a DataFrame containing the .csv data. #
df = pd.read_csv("../data/mother_daughter_heights.csv")

# Extract the data into NumPy arrays #
mothers_X = np.array(df['mother'])
daughters_Y = np.array(df['daughter'])
