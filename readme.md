# Requirements

In order to run the code in this repository, you will need to install the following packages:
* Numpy
* Scipy
* Matplotlib
* sklearn

# Running the code
In order to run the code, you will need to open the main.ipynb (jupyter) notebook file and run the cells in order. You are required to run the cells in order, as the code is dependent on the output of the previous cells.

To provide data to the code, you will need to place the data in a folder named "data" located in the same location as the main.ipynb file.
The file structure required for the code to run is as follows:
* main.ipynb
* main.py (optional, used to create distribution plots per feature)
* pointcloud.py
* test_plot.py
* train_py
* /data
    * /pointclouds
        * 000.xyz
        * 001.xyz
        * 002.xyz
        * ...
    
Please note that the code requires the provided .xyz to be numbered 0 to 499,
and every 100 points need to be point clouds of a certain entity categorized in the following way:
* 0-99: building
* 100-199: car
* 200-299: fence
* 300-399: pole
* 400-499: tree