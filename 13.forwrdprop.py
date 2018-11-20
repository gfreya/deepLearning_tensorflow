import numpy as np

x = [1,2]
state = [0.0,0.0]
w_cell_state = np.asarray([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
b_cell = np.asarray([0.1,-0.1])
w_output = np.asarray([[1.0],[2.0]])
b_output = 0.1

for i in range(len(X)):
    state = np.append(state,X[i])
    before_activation = np.dot(state,w_cell_state)+b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state,w_output)+b_output
    print("state_value_%i: "%i,state)
    print("output_value_%i: "%i,final_output)