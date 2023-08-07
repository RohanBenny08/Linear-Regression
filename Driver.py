from LinearRegressor import LinearRegressor
from LogisticRegressor import LogisticRegressor
import os
import matplotlib.pyplot as plt

#import right packages for doing plotting

def sensorLocPredictionErrorHistogram(lin1_y_test_pred_err):

    #do plotting of histogram here

    plt.hist(lin1_y_test_pred_err)
    plt.xlabel('Prediction values')
    plt.ylabel('Data points')
    plt.savefig("SensorLocLinearRegressionError.pdf")
    plt.clf()

def linearAndLogisticRegressionErrorComparison(lin_y_test_pred_err,log_y_test_pred_err):
    # do plotting here
    plt.hist(lin_y_test_pred_err)
    plt.xlabel('Regression error')
    plt.ylabel('Data points')
    plt.savefig("WineQualityLinearRegressionError.pdf")
    plt.clf()
    
    # do plotting here
    plt.hist(log_y_test_pred_err)
    plt.xlabel('Regression error')
    plt.ylabel('Data points')
    plt.savefig("WineQualityLogisticRegressionError.pdf")
    plt.clf()

def visualizeChlorineSulphateScatterPlot(X,path):
    chlorineData = X[:,4]
    sulphateData = X[:,9]

    # do plotting here
    plt.scatter(chlorineData,sulphateData)
    plt.xlabel('chlorineData')
    plt.ylabel('sulphateData')
    plt.savefig(path)
    plt.clf()

def visualizeCitricAcidBoxPlot(X,path):
    citricAcidData = X[:, 2]

    # do plotting here
    plt.boxplot(citricAcidData)
    plt.xlabel('citricAcidData')
    plt.ylabel('Datapoint')
    plt.savefig(path)
    plt.clf()

def visualizeDensityHistogram(X,path):
    densityData = X[:, 7]

    # do plotting here
    plt.hist(densityData)
    plt.xlabel('Density')
    plt.ylabel('Data points')
    plt.savefig(path)
    plt.clf()

trainfilename = os.path.join("data","sensor_loc_train.csv")
testfilename = os.path.join("data","sensor_loc_test.csv")

alpha = 0.05
linr = LinearRegressor("sensor_loc", alpha, trainfilename, testfilename)
lin1_y_test_pred_err = linr.evaluateRegressor()

trainfilename = os.path.join("data","wine_quality_train.csv")
testfilename = os.path.join("data","wine_quality_test.csv")

alpha = 0.05
linr = LinearRegressor("wine_quality", alpha, trainfilename, testfilename)
lin_y_test_pred_err = linr.evaluateRegressor()

logr = LogisticRegressor("wine_quality", alpha, trainfilename, testfilename)
log_y_test_pred_err = logr.evaluateRegressor()

X,y = LinearRegressor.readData(trainfilename)
visualizeChlorineSulphateScatterPlot(X, path="example_images/ChlorineSulphateScatterPlot.pdf")
visualizeCitricAcidBoxPlot(X, path="example_images/CitricAcidBoxPlot.pdf")
visualizeDensityHistogram(X, path="example_images/DensityHistogram.pdf")

sensorLocPredictionErrorHistogram(lin1_y_test_pred_err)
linearAndLogisticRegressionErrorComparison(lin_y_test_pred_err,log_y_test_pred_err)