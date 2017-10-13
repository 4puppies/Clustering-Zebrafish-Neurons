# Clustering-Zebrafish-Neurons

This project takes simulated electrical impulses in the brain of a zebrafish and uses that data to categorize neurons into 19 different clusters, analogous to the 19 differents sections in a zebrafish's brain. The techniques used were K-Means clustering, a machine learning algorithm, and low pass filtering, a concept from digital signal processing. 

## Getting Started

### Prerequisites

You're going to need to have Python installed on your computer, as well as ```pip```/```pip3``` in order to install the scipy, sklearn, matplotlib, and numpy libraries. Instructions for installing Python and pip are found [here](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation).

Once you have Python and pip, go ahead and use pip to download the following:

* sklearn
* scipy
* numpy
* matplotlib

These are popular machine learning libraries, so it's a must-have if you will be frequently using Python for machine learning. 

### Deployment

Go ahead and download ```sample_data.zip```, which is a small sample of electrical impulse data for 1000 neurons within the zebrafish brain. Then go ahead and download ```NeuronClustering.py```. Make sure that the unzipped file and the Python file are both saved in the same directory or else it won't run.

And that's it! 

### Notes

The accompanying research paper was with a much larger dataset than the one provided here, so the results are a little different, but the logic is exactly the same.

### Acknowledgements 

This wouldn't have been possible without [Kaggle](https://www.kaggle.com/), as the concept for this project is directly related to a [competition they posted online](https://www.kaggle.com/c/connectomics). 
