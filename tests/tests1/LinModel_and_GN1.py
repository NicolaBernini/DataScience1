# 
# @brief Linear Model and Gaussian Noise Example 
# 

import numpy as np
from pymc3 import *
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
%matplotlib inline

class LinModel: 
    q = 0 # Default Value 
    m = 1 # Default Value 
    def __init__(self, in_m, in_q) : 
        self.m = in_m 
        self.q = in_q
    def get(self, x): # Generates the Output Corresonding to the Input 
        return self.q + self.m*x # Linear Model 

class Noise: 
    @staticmethod 
    def get_Normal(in_scale, in_size): 
        return np.random.normal(scale=in_scale, size=in_size)

class out_y : 
    true = 0 # The GT 
    noisy1 = 0 
    noisy3 = 0 
    
class db_t : 
    size = 10 
    model = LinModel(10,1) # True Model is Linear 
    x = 0 # Domain 
    y = out_y() # Codomain 
    
    def __init__(self): 
        self.x = np.linspace(0, 1, self.size) # In constructor build domain 

db = db_t()

db.y.true = db.model.get(db.x) # Get GT for Domain 
print(db.y.true)

db.y.noisy1 = db.y.true + Noise.get_Normal(0.5, db.size) # Get Actual Noisy Observation 
db.y.noisy3 = db.y.true + Noise.get_Normal(0.8, db.size) # Get Actual Noisy Observation 

print(db.y.noisy1)
print(db.y.noisy3)


# Note: We expected that the smaller the noise the smaller the MAE hence the better the model wrt GT appunto 
print(mean_absolute_error(db.y.true,db.y.noisy1)) # < Print GT vs Noise1 Metric appunto 
print(mean_absolute_error(db.y.true,db.y.noisy3)) # < Print GT vs Noise3 Metric appunto 

# Note: With balanced average we expect an improvement wrt to single noisy signals as a result of partial noise cancellations appunto 
w = 0.5
print("Straight Average = " + str(mean_absolute_error(db.y.true, db.y.noisy1*w + db.y.noisy3*(1-w)))) # < A balanced weight average appunto 

# Note: With weighted average we expect another improvement wrt balanced average as we perform better cancellation adjusting noise weights appunto 
w = 0.7
print("Weighted Average = " + str(mean_absolute_error(db.y.true, db.y.noisy1*w + db.y.noisy3*(1-w)))) # < A weighted average 



# add noise
#model1 = true_regression_line + np.random.normal(scale=.5, size=size) #Noisy
#model2 = true_regression_line + np.random.normal(scale=.2, size=size) #Less Noisy
