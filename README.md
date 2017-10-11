# Clustering-Zebrafish-Neurons

This project takes simulated electrical impulses in the brain of a zebrafish and uses that data to categorize neurons into 19 different clusters, analogous to the 19 differents sections in a zebrafish's brain. The techniques used were K-Means clustering, a machine learning algorithm, and low pass filtering, a concept from digital signal processing. 

## Getting Started

### Prerequisites

You're going to need to have Python installed on your computer, as well as ```pip```/```pip3``` in order to install the scipy, sklearn, matplotlib, and numpy libraries. Instructions for installing Python and pip are found [here](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation).

Once you have Python and pip, go ahead and use pip download the following:

* sklearn
* scipy
* numpy
* matplotlib

These are popular machine learning libraries, so it's a must-have if you will be frequently using Python for machine learning. 

### Deployment

Go ahead and download ```small_sample.txt```, which is a small sample of electrical impulse data for 1000 neurons within the zebrafish brain. Then go ahead and download ```NeuronClustering.py```. Make sure that these files are both saved in the same directory or else it won't run.

And that's it! 

I commented out a lot of the "debugging" graphs that I used just to do a sanity check on what the data should look like. The final graph however, the one that represents the number of neurons in a given cluster, is not commented out and that should appear fine when you run the program. 

### Acknowledgements

This wouldn't have been possible without [Kaggle](https://www.kaggle.com/), as the concept for this project is directly related to a [competition they posted online](https://www.kaggle.com/c/connectomics). 
