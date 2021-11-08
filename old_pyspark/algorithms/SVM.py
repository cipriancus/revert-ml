from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *


class SVMClassifier:

    def __init__(self, data, features, labelCol, maxIter=100, regParam=0.0, tol=1e-6, threshold=0.0,
                 aggregationDepth=2):
        self.features = features

        self.labelCol = labelCol

        self.settings = [('maxIter', maxIter), ('regParam', regParam), ('tol', tol), ('threshold', threshold),
                         ('aggregationDepth', aggregationDepth)]

        vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")

        self.SVM = LinearSVC(maxIter=maxIter, regParam=regParam, tol=tol, threshold=threshold,
                             aggregationDepth=aggregationDepth, labelCol=labelCol)

        pipeline = Pipeline(stages=[vectorAssembler, self.SVM])

        self.model = pipeline.fit(data)

    def classify_testdata(self, validate):
        lr_data = validate.select(col(self.labelCol).alias("label"), *self.features)

        prediction = self.model.transform(lr_data)

        return prediction

    def classification_report(self, predict, validate):
        predict_list = [i.prediction for i in predict.select("prediction").collect()]
        test_class = [i[self.labelCol] for i in validate.select(self.labelCol).collect()]  # self.test_file['Class']
        print("############################SVM############################")
        print('SVM  with settings ' + str(self.settings))
        print(classification_report(test_class, predict_list))
        print("Confusion Matrix for SVM")
        print(confusion_matrix(test_class, predict_list))
        print("##########################################################")
