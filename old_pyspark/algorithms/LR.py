from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *


class LR:

    def __init__(self, data, features, labelCol):
        self.features = features

        self.labelCol = labelCol

        vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")

        self.lr = LogisticRegression(labelCol=labelCol)

        stages = [vectorAssembler, self.lr]

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
        print("############################LR############################")
        print('Accuracy for LR')
        print(classification_report(test_class, predict_list))
        print("Confusion Matrix for LR")
        print(confusion_matrix(test_class, predict_list))
        print("##########################################################")
