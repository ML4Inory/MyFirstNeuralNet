import pickle
import numpy as np
import pprint

pkl = open('data.pkl','rb')
w1 = pickle.load(pkl)
pprint.pprint(w1)
# print w1