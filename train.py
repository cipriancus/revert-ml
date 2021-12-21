import csv
from sklearn import svm
from sklearn.metrics import classification_report


def LoadData(locFile):
    with open(locFile) as csv_file:
        data = []
        label = []
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append(row)
            line_count += 1
        for row in data:
            if row[20] == '0':
                label.append(0)
            else:
                label.append(1)
            del row[20]
        return data, label, line_count


locFileTrain = "crc_msk_2017_clinical_data.tsv"  # load data train
total_file, totalLabel_file, line_count = LoadData(locFileTrain)

train_number = round(line_count * 0.2)

test, testLabel = total_file[0:train_number], totalLabel_file[0:train_number]
train, trainLabel = total_file[train_number:], totalLabel_file[train_number:]

# clf = svm.SVC(kernel='linear')
# clf.fit(train, trainLabel)
# testAcc = clf.score(test, testLabel)
#
# for i in range(len(testLabel)):  # transform label to before discrititation
#     if testLabel[i] == 1:
#         testLabel[i] = "POSITIVE"
#     else:
#         testLabel[i] = "NEGATIVE"
#
# predict = []
# for i in range(len(test)):
#     result = clf.predict([test[i]])
#     if result == [1]:
#         predict.append("POSITIVE")
#     else:
#         predict.append("NEGATIVE")
#
# print(classification_report(testLabel, predict))
#
# #   TP FN
# #   FP TN
# TP = 0  # True positive
# FN = 0  # False negative
# FP = 0  # False positive
# TN = 0  # True negative
#
# for i in range(len(predict)):
#     if testLabel[i] == "POSITIVE" and predict[i] == "POSITIVE":
#         TP += 1
#     if testLabel[i] == "POSITIVE" and predict[i] == "NEGATIVE":
#         FN += 1
#     if testLabel[i] == "NEGATIVE" and predict[i] == "POSITIVE":
#         FP += 1
#     if testLabel[i] == "NEGATIVE" and predict[i] == "NEGATIVE":
#         TN += 1
#
# print("True Positive (TP) :", TP)
# print("False Negative (FN) :", FN)
# print("False Positive (FP) :", FP)
# print("True Negative (TN) :", TN)
#
# accuracy = testAcc
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# fMeasure = 2 * precision * recall / (precision + recall)
# print("Accuracy :", round(testAcc, 4) * 100, "%")
# print("Precision :", round(precision, 4) * 100, "%")
# print("Recall :", round(recall, 4) * 100, "%")
# print("F-1 Measure :", round(fMeasure, 4) * 100, "%")

import pickle
# # save
# with open('model.pkl','wb') as f:
#     pickle.dump(clf,f)

# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict([test[2]]))