#%% readme!
# 1. Please make sure the "audio" directory exists in the "dsl_data" directory.
# 2. In the network, Torch library uses CUDA. Therefore, please make sure that
#    CUDA is available. Otherwise, training takes years to get done!
# 3. The output of this code is a graph illustrating accuracy vs. epoch during
#    training and test. The graph is also stored in the current folder named
#    "resultDevelopment.png". Also, a CSV file is generated that contains the
#    result of applying the trained model to the evaluation dataset. The file is
#    named "resultEvaluation.csv" and stored in the current folder.

#%% importing functions
import extraFunctions

#%% setting the base dir
folderBase = extraFunctions.getBase()

#%% installing the required libraries
extraFunctions.installReq(folderBase)

#%% preparation
extraFunctions.preparation(folderBase)

#%% prerocessing
extraFunctions.preprocessDev(folderBase)
extraFunctions.preprocessEval(folderBase)

#%% development
extraFunctions.develop(folderBase)

#%% evaluation
extraFunctions.evaluate(folderBase)

#%% plot
extraFunctions.plotResult(folderBase)
