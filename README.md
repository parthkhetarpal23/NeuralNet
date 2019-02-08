# NeuralNet
One of the assignments in my ML Course at UT Dallas. 
Before executing: 
Paste the absolute path of the .csv files (datasets) used in the code.

Replace “relu” with “sigmoid” or “tanh” in the following lines of the code to use the particular activation function. 

line 296:	testError = neural_network.predict(X_test, y_test, activation="relu")   #main function 
line 157:	out = self.forward_pass(activation="relu")       #df train function 
line 159:	self.backward_pass(out, activation="relu")     #df train function 

To execute the code: 
Run ‘NeurakNet.py’ file on any python IDE .

Run via Cmd:
<path> python3 driver.py 
