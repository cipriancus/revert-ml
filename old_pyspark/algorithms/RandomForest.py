from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *


class RFClassifier:

    def __init__(self, data, features, labelCol, maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                 maxMemoryInMB=256, impurity="gini", numTrees=20):
        self.features = features

        self.labelCol = labelCol

        self.settings = [('maxDepth', maxDepth), ('maxBins', maxBins), ('minInstancesPerNode', minInstancesPerNode),
                         ('minInfoGain', minInfoGain), ('maxMemoryInMB', maxMemoryInMB), ('impurity', impurity),
                         ('numTrees', numTrees)]

        vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")

        self.nb = RandomForestClassifier(maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode,
                                         minInfoGain=minInfoGain,
                                         maxMemoryInMB=maxMemoryInMB, impurity=impurity, numTrees=numTrees,
                                         labelCol=labelCol)

        stages = [vectorAssembler, self.nb]

        pipeline = Pipeline(stages=stages)

        self.model = pipeline.fit(data)

    def classify_testdata(self, validate):
        lr_data = validate.select(col(self.labelCol).alias("label"), *self.features)

        prediction = self.model.transform(lr_data)

        return prediction

    def classification_report(self, predict, validate):
        """
        Function that computes confusion matrix to evaluate the accuracy of the classification
        :param predict: The predicted labels that is used to compute the confusion matrix
        :return: The confusion matrix
        """
        predict_list = [i.prediction for i in predict.select("prediction").collect()]
        test_class = [i[self.labelCol] for i in validate.select(self.labelCol).collect()]  # self.test_file['Class']
        print("############################RF############################")
        print('Accuracy for RF with settings ' + str(self.settings))
        print(classification_report(test_class, predict_list))
        print("Confusion Matrix for RF")
        print(confusion_matrix(test_class, predict_list))
        print("##########################################################")
