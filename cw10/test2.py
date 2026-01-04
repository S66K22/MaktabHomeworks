from test import model
import pickle

clf = model(kernel='linear')

with open('path/r.pkl', 'w') as f:
    f.write('s')