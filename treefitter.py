
import pandas as pd
from datapredictionmachine import DataPredictionMachine

test_data_1_frame = pd.read_csv('./data/party/1.csv')
test_data_2_frame = pd.read_csv('./data/party/2.csv')
test_data_3_frame = pd.read_csv('./data/party/3.csv')

train_data_12_frame = pd.read_csv('./data/party/12.csv')
train_data_13_frame = pd.read_csv('./data/party/13.csv')
train_data_23_frame = pd.read_csv('./data/party/23.csv')

trainer = DataPredictionMachine("overfitting-", train_data_12_frame, False)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values()
trainer.create_dummy_vars()
trainer.clean()
trainer.preview()
trainer.generate_tree(None) # no max depth

# -----------------------------------------------------
# CALC SCORE WITH THE NOT TRAINED SET
# -----------------------------------------------------


tester = DataPredictionMachine("overfitting-", test_data_3_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
score = trainer.calc_score(tester.get_dataframe())
print("Model score (train data):", score)






#-------------------------------------------------------

trainer = DataPredictionMachine("overfitting-", train_data_13_frame, False)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values()
trainer.create_dummy_vars()
trainer.clean()
trainer.preview()
trainer.generate_tree(None) # no max depth

# -----------------------------------------------------
# CALC SCORE WITH THE NOT TRAINED SET
# -----------------------------------------------------


tester = DataPredictionMachine("overfitting-", test_data_2_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
score = trainer.calc_score(tester.get_dataframe())
print("Model score (train data):", score)






#-------------------------------------------------------

trainer = DataPredictionMachine("overfitting-", train_data_23_frame, False)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values()
trainer.create_dummy_vars()
trainer.clean()
trainer.preview()
trainer.generate_tree(None) # no max depth

# -----------------------------------------------------
# CALC SCORE WITH THE NOT TRAINED SET
# -----------------------------------------------------


tester = DataPredictionMachine("overfitting-", test_data_1_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
score = trainer.calc_score(tester.get_dataframe())
print("Model score (train data):", score)



