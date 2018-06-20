import pandas as pd
from datapredictionmachine import DataPredictionMachine

# -----------------------------------------------------
# LOAD TRAIN DATA, PREVIEW, CLEAN, GENERATE TREE & LOG
# -----------------------------------------------------
train_data_frame = pd.read_csv('train.csv')
trainer = DataPredictionMachine(train_data_frame, "output", True)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values()
trainer.clean()
trainer.preview()
trainer.generate_tree()

# -----------------------------------------------------
# CALC SCORE WITH THE TRAINED SET
# -----------------------------------------------------
score = trainer.calc_score(trainer.get_dataframe())
print("Model score (train data):", score)

# -----------------------------------------------------
# LOAD THE TEST DATA, TRANSFORM & CLEAN
# -----------------------------------------------------
test_data_frame = pd.read_csv('test.csv')
tester = DataPredictionMachine(test_data_frame, None, False)
tester.prepare()
tester.handle_missing_values()
tester.clean()

# -----------------------------------------------------
# PREDICT SURVIVAL FOR TEST DATA
# -----------------------------------------------------
predict_survival = trainer.predict(tester.get_dataframe())
print("Predicted survival for test data:\n", predict_survival)

# -----------------------------------------------------
# SCORE AGAINST COMPLETE DATASET
# -----------------------------------------------------
# test_data_frame2 = pd.read_csv('test2.csv')
# tester2 = DataPredictionMachine(test_data_frame2, None, False)
# tester2.prepare()
# tester2.handle_missing_values()
# tester2.clean()
# score2 = trainer.calc_score(tester2.get_dataframe())
# print("Model score (test2 data):", score2)
