# 
# @brief Linear Model and Gaussian Noise Example 
# 

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

# add noise
#model1 = true_regression_line + np.random.normal(scale=.5, size=size) #Noisy
#model2 = true_regression_line + np.random.normal(scale=.2, size=size) #Less Noisy