
import pandas as pd
from datapredictionmachine import DataPredictionMachine

test_data_1_frame = pd.read_csv('./data/party/1.csv')
test_data_2_frame = pd.read_csv('./data/party/2.csv')
test_data_3_frame = pd.read_csv('./data/party/3.csv')

train_data_12_frame = pd.read_csv('./data/party/1u2.csv')
train_data_13_frame = pd.read_csv('./data/party/1u3.csv')
train_data_23_frame = pd.read_csv('./data/party/2u3.csv')

trainer = DataPredictionMachine("overfitting-1", train_data_12_frame, False)
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


tester = DataPredictionMachine("overfitting-1", test_data_3_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
score1 = trainer.calc_score(tester.get_dataframe())







#-------------------------------------------------------

trainer = DataPredictionMachine("overfitting-2", train_data_13_frame, False)
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


tester = DataPredictionMachine("overfitting-2", test_data_2_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
score2 = trainer.calc_score(tester.get_dataframe())






#-------------------------------------------------------

trainer = DataPredictionMachine("overfitting-3", train_data_23_frame, False)
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


tester = DataPredictionMachine("overfitting-3", test_data_1_frame, False)
tester.prepare()
tester.handle_missing_values()
tester.create_dummy_vars()
tester.clean()
score3 = trainer.calc_score(tester.get_dataframe())






print("Model scores:\n Model 1: "+str(score1)+"\n Model 2: "+str(score2)+"\n Model 3: "+str(score3))



