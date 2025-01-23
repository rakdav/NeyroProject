import numpy as np
from tensorflow.python.keras.backend import learning_phase

ratings = np.array([
    [5,3,0,1],
    [4,0,0,1],
    [1,1,0,5],
    [1,0,0,4],
    [0,1,5,4]
])
num_users,num_items=ratings.shape
num_factors=2
learning_rate=0.01
num_epoches=1000

user_matrix=np.random.rand(num_users,num_factors)
item_matrix=np.random.rand(num_factors,num_items)

