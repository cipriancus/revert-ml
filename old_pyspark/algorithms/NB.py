from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *


class NbClassifier:

    def __init__(self, data, features, labelCol, smoothing=1.0, modelType="multinomial"):
        self.features = features

        self.labelCol = labelCol

        self.settings = [('smoothing', smoothing), ('modelType', modelType)]

        vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")

        self.nb = NaiveBayes(smoothing=smoothing, modelType=modelType, labelCol=labelCol)

        pipeline = Pipeline(stages=[vectorAssembler, self.nb])

        self.model = pipeline.fit(data)

    def classify_testdata(self, validate):
        lr_data = validate.select(col(self.labelCol).alias("label"), *self.features)

        prediction = self.model.transform(lr_data)

        return prediction

    def classification_report(self, predict, validate):
        predict_list = [i.prediction for i in predict.select("prediction").collect()]
        test_class = [i[self.labelCol] for i in validate.select(self.labelCol).collect()]  # self.test_file['Class']
        print("############################NB############################")
        print('NB  with settings ' + str(self.settings))
        print(classification_report(test_class, predict_list))
        print("Confusion Matrix for NB")
        print(confusion_matrix(test_class, predict_list))
        print("##########################################################")
