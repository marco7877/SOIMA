 This code trains a Self-Organizing Internal Model Architecture (SOIMA);
 after being trained, weights may be saved as .txt  To plot the SOM, this script search for the winner nodes for each element from
 the training element (normalized in range 0:1), then it determines which type
 of data such node stands for by the most common element (label) identified with
 such node. Thus, que plotting still works for testing data.
#**hebbianian learning only works with squared modal and amodal soms***
