import pandas as pd
import numpy as np

### least_squares_fit(X,Y) ###
# params
#   X -> Array containing the predictor variables.
#   Y -> Array containing the predicted variables.
#
# return
#   beta_0 -> The y-intercept model parameter.
#   beta_1 -> The slope model parameter.
###
def least_squares_fit(X,Y):
    # Let's get the means for each variable #
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)
    # We need to sum the product of the difference of each X & Y's elements from their mean. #
    Xdiff = X-X_bar
    Ydiff = Y-Y_bar
    # For the numerator we need to multiply the differences in each variable #
    numerator = np.sum( Xdiff*Ydiff )
    # For the denominator we need to square the differences in X #
    denominator = np.sum( Xdiff**2 )
    # We have everything to calculate our model parameters. #
    beta_1 = numerator/denominator
    beta_0 = Y_bar-beta_1*X_bar
    return beta_0,beta_1
