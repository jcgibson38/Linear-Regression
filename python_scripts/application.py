import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import least_squares_fit

### predict###
# params
#   X -> Array of predictor values.
#   beta_0,beta_1 -> Model parameters.
# return
#   Array of predicted values based on model parameters.
###
def predict(X,beta_0,beta_1):
    return X*beta_1+beta_0

# Create a DataFrame containing the .csv data. #
df = pd.read_csv("../data/mother_daughter_heights.csv")

# Extract the data into NumPy arrays #
mothers_X = np.array(df['mother'])
daughters_Y = np.array(df['daughter'])

# Apply linear regression to data #
beta_0,beta_1 = least_squares_fit(mothers_X,daughters_Y)

# Let's plot our data and the model #
xs = np.array( range(int(np.min(mothers_X))-3,int(np.max(mothers_X))+3) )
plt.plot(mothers_X,daughters_Y,'bo')
plt.plot( xs,predict(xs,beta_0,beta_1),'r' )

# Set some plot labels #
plt.xlabel('Mothers Height (in.)')
plt.ylabel('Daughters Height (in.)')

plt.show()
