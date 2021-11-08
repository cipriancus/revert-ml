from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import utils.BuildDataset as buildDataset
from algorithms.NB import NbClassifier
from algorithms.RandomForest import RFClassifier
from algorithms.GBT import GBT
from algorithms.DT import DT
from algorithms.LR import LR
from algorithms.SVM import SVMClassifier
import warnings
import os

os.chdir("data/")

# "metastasectomy_site",
# "first_site_of_metastasis",

features = ["age",
            "chemo_exposure",
            "fraction_genome_altered",
            "gene_panel",
            "mCRC_type",
            "metastasectomy",
            "metastases_site_first_bone",
            "metastases_site_first_brain",
            "metastases_site_first_gynecological",
            "metastases_site_first_liver",
            "metastases_site_first_ln",
            "metastases_site_first_lung",
            "metastases_site_first_pelvis",
            "metastases_site_first_peritoneum_omentum_abdomen",
            "metastatic_biopsy_site",
            "msi_score",
            "msi_status",
            "mutation_count",
            "overall_survival_months",
            "living_status",
            "other_metastasis_sites",
            "patient_tumor_grade",
            "primary_tumor_site",
            "primary_tumor_location",
            "sample_type",
            "sex",
            "specimen_type",
            "stage_at_diagnostic",
            "time_from_met_dx_to_sequencing",
            "time_to_81_months"]

labelCol = 'living_status'

warnings.filterwarnings("ignore")

conf = SparkConf().setAppName("revert")

conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '12G')
        .set('spark.driver.memory', '12G')
        .set('spark.driver.maxResultSize', '12G'))

spark_context = SparkContext(conf=conf)

sqlContext = SQLContext(spark_context)

data = buildDataset.read_dataset(sqlContext)

data.cache()

train, validate = data.randomSplit([0.1, 0.9], seed=12345)

print((train.count(), len(train.columns)))
print((validate.count(), len(validate.columns)))

print('---- 1 -----')
print('Naive Bayes Classifier')
nb_classifier = NbClassifier(train, features, labelCol)
prediction = nb_classifier.classify_testdata(validate)
nb_classifier.classification_report(prediction, validate)
del nb_classifier

print('---- 2 -----')
print('Random Forest Classifier')
rf_classifier = RFClassifier(train, features, labelCol)
prediction = rf_classifier.classify_testdata(validate)
rf_classifier.classification_report(prediction, validate)
del rf_classifier

print('---- 3 -----')
print('Decision Tree Classifier')
dt_classifier = DT(train, features, labelCol)
prediction = dt_classifier.classify_testdata(validate)
dt_classifier.classification_report(prediction, validate)
del dt_classifier

print('---- 4 -----')
print('Gradient Boosted Trees')
gbt_classifier = GBT(train, features, labelCol)
prediction = gbt_classifier.classify_testdata(validate)
gbt_classifier.classification_report(prediction, validate)
del gbt_classifier

print('---- 5 -----')
print('Logistic Regression')
lr_classifier = LR(train, features, labelCol)
prediction = lr_classifier.classify_testdata(validate)
lr_classifier.classification_report(prediction, validate)
del lr_classifier

print('---- 6 -----')
print('SVM')
svm_classifier = SVMClassifier(train, features, labelCol)
prediction = svm_classifier.classify_testdata(validate)
svm_classifier.classification_report(prediction, validate)
del svm_classifier
