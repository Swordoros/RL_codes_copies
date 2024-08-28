import pickle
def load_Q_from_file(filename='P_data.txt'):
    with open(filename, 'rb') as file:
        Q = pickle.load(file)
    return Q

Q = load_Q_from_file()
print(Q)