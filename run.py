import pandas as pd
from datapredictionmachine import DataPredictionMachine

create_describing_images = False

# -----------------------------------------------------
# LOAD TRAIN DATA, UNDERSTANDING, PREPERATION, MODELING (TREE)
# -----------------------------------------------------
print("\n-------------------------------------- TRAIN MODEL ------------------------------------------\n")
train_data_frame = pd.read_csv('./data/train.csv')
trainer = DataPredictionMachine("train-1", train_data_frame, True)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values()
if(create_describing_images):
    trainer.create_describing_images()
trainer.create_dummy_vars()
trainer.clean()
trainer.preview()
trainer.generate_tree(8)
# -----------------------------------------------------
# CALC SCORE WITH THE TRAINED SET
# -----------------------------------------------------
score = trainer.calc_score(trainer.get_dataframe())
print("Model score (train data):", score)

# -----------------------------------------------------
# LOAD THE TEST DATA, PREPARATION, PREDICTION
# -----------------------------------------------------
print("\n-------------------------------------- PREDICT TESTDATA ------------------------------------------\n")
test_data_frame = pd.read_csv('./data/test.csv')
tester = DataPredictionMachine("test-1", test_data_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
# -----------------------------------------------------
# PREDICT SURVIVAL FOR TEST DATA
# -----------------------------------------------------
predict_survival = trainer.predict(tester.get_dataframe())
print("Predicted survival for test data:\n", predict_survival)
solution = pd.read_csv('./data/gender_submission.csv')
solution = solution['Survived'].values
trainer.evaluate(predict_survival, solution)





# -----------------------------------------------------
# SCORE AGAINST COMPLETE DATASET
# -----------------------------------------------------
# test_data_frame2 = pd.read_csv('./data/test2.csv')
# tester2 = DataPredictionMachine("test-2", test_data_frame2, False)
# tester2.prepare()
# tester2.handle_missing_values()
# tester2.clean()
# score2 = trainer.calc_score(tester2.get_dataframe())
# print("Model score (test2 data):", score2)
