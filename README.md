# K-means_Algorithm
## Background
Lightning networks detect individual lightning discharge events all over the world. For each lightning event, the data is stored as (time, amount of discharge, latitude, and longitude). It shows that when and where, there is a lightening stroke with certain amount of discharge. Clustering helps the meteorologists to group the lightning strokes basing on the position. 

2010825.txt records the places of the lightning strokes during 10:00-23:59pm on the day 25/08/2010.

Below is the plot of the data.

![image](https://user-images.githubusercontent.com/109471364/179456464-c6ebbcd2-0d4f-42bc-8399-d68418dee24d.png)

The dataset in the file only contains two features for clustering:
1. X: x-position in coordinate system, which is converted from latitude.
2. Y: y-position in coordinate system, which is converted from longitude.

NOTE: a test.txt with 15 objects is also provided to help you verify your code. 

I suggest implement and test your code on test.txt first and then apply the code on lightening data.

## Tasks
a. Write a function loadData(filename) to load lightening data into an array X . After appending the third column for storing the clusterID . X.shape should be (16259,3).

b. Plot the data in the given dataset to visualize the input of the clustering. Paste your plot in Section 1 Lighting data of your report.

c. Construct the clustering objective function for K-Means. Briefly explain this equation. Write down them in Section 2 Objective Function of your report. Note: In this assignment, you will use Euclidean distance metric in the objective function.

d. Use the objective function (from task c) to write a function errCompute() for evaluating the quality of clustering. This function takes in dataset X and means M (with shape (K,2)), returns the mean of the error J . Suppose that the cluster id for all objects in X are initialized as 0, and M = np.array([[0,0]]) , run errCompute(X,M) , you should get a result of 86.62.

e. Write down and briefly explain the method of assigning each object to a cluster in Section 3 assigning objects of your report.

f. Use the assigning method from task e to implement function Group() : to assign each object into a cluster basing on the current set of means M . This function takes in dataset X , M the current means. It returns the clustering result included in X . Suppose K=5, and the top 5 objects from X are chosen as initial means. i.e. M=np.copy(X[0:5,0:X.shape[1]-1]) . After running X= Group(X,M) , you should get errCompute(X,M) = 9.73.

g. Describe the method of calculating new means basing on current clustering in Section 4 Updating Means of your report.

h. Use the method described in task g to implement the function calcMean(). This function takes in dataset X, M the current means and returns the updated M. K is still set as 5, repeatedly run calcMeans() and Group() until there is no changes in clustering result, you’ll get errCompute(X,M)=4.01. The clustering result (k=5) shown below:

![image](https://user-images.githubusercontent.com/109471364/179457207-6fa9e074-fda1-4a6c-ba73-0f9b23f80ad8.png)

i. Run your k-means when K is chosen as 50 and 100 respectively and plot the clustering result. Compare the objective function values (i.e. value returned by errCompute() ) for the two clustering results, and discuss which one is a better result. Write down you discussion in Section 5 Choosing K.

## Script
- kmeans
