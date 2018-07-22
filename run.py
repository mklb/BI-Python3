import pandas as pd
from datapredictionmachine import DataPredictionMachine
from helper import CostBenefitMatrix
import sys

# settings that can be changed
use_log_file = False                    # <- set True to create a log file (will be overwritten each time)
create_describing_images = False        # <- set True to create describing images (creating images will take quite long)
create_missing_value_matrix = False     # <- set True to create missing value matrices each time before and after handling missing values (will spam the output) (at least one is drawn each run)
log_output_training = True              # <- set True to see all logs during training
log_output_testing = False              # <- set True to see all logs during training
cost_benefit = CostBenefitMatrix(5, 2000)   # <- determine cost-benefit-matrix (price, loss)

if(use_log_file):
    print("\n\n---------------------\nEverything will be written to messages.log.\n--------------------\n\n")
    old_stdout = sys.stdout
    log_file = open("log-file.log","w")
    sys.stdout = log_file

print("Logging run:\n\n")

print("settings:")
print("create_describing_images = " + str(create_describing_images))
print("create_missing_value_matrix = " + str(create_missing_value_matrix))
print("log_output_training = " + str(log_output_training))
print("log_output_testing = " + str(log_output_testing))
print("\n")


# -----------------------------------------------------
# LOAD TRAIN DATA, UNDERSTANDING, PREPERATION, MODELING (TREE)
# tree will be saved in ./output/train-1
# -----------------------------------------------------
print("\n-------------------------------------- TRAIN MODEL ------------------------------------------\n")
train_data_frame = pd.read_csv('./data/train.csv')
trainer = DataPredictionMachine("train-1", train_data_frame, log_output_training)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values(create_missing_value_matrix)
if(create_describing_images):
    trainer.create_describing_images()
trainer.create_dummy_vars()
trainer.clean()
trainer.preview()
trainer.generate_tree(6)
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
tester = DataPredictionMachine("test-1", test_data_frame, log_output_testing)
tester.prepare()
tester.handle_missing_values(create_missing_value_matrix)
tester.create_dummy_vars()
tester.clean()
tester_data = tester.get_dataframe()
predict_survival_tester = trainer.predict(tester_data)
print("Predicted survival for test data:\n", predict_survival_tester)
# -----------------------------------------------------
# EVALUATE MODEL FOR TRAIN DATA
# -----------------------------------------------------
trainer_data = trainer.get_dataframe()
cost_benefit_info = cost_benefit.get_cost_benefit_matrix_info()
solution = trainer_data['Survived'].values
predict_survival_trainer = trainer.predict(trainer_data.iloc[:, 1:])
expected_rates_trainer = trainer.evaluate(predict_survival_trainer, solution)
expected_profit_trainer = trainer.calc_expected_profit(expected_rates_trainer, cost_benefit_info)
print("Expected Profit: " + str(expected_profit_trainer))

# -----------------------------------------------------
# COMPARISON MODEL "SEX"
# -----------------------------------------------------
print("\n-------------------------------------- COMPARISON MODEL 'SEX' ------------------------------------------\n")
train_data_frame = pd.read_csv('./data/train.csv')
cost_benefit_info = cost_benefit.get_cost_benefit_matrix_info()
comparer_sex = DataPredictionMachine("train-1", train_data_frame, False)
comparer_sex.prepare()
comparer_sex.handle_missing_values(create_missing_value_matrix)
comparer_sex.create_dummy_vars()
comparer_sex.clean()
comparer_sex.generate_tree(1)
comparer_sex_data = comparer_sex.get_dataframe()
solution = comparer_sex_data['Survived'].values
predict_survival_comparer_sex = comparer_sex.predict(comparer_sex_data.iloc[:, 1:])
expected_rates_comparer_sex = comparer_sex.evaluate(predict_survival_comparer_sex, solution)
expected_profit_comparer_sex = comparer_sex.calc_expected_profit(expected_rates_comparer_sex, cost_benefit_info)
print("Expected Profit: " + str(expected_profit_comparer_sex))

if(expected_profit_trainer > expected_profit_comparer_sex):
    print("\n\n\n The developed modell is better than gender-only baseline!")




if(use_log_file):
    sys.stdout = old_stdout
    log_file.close()
    print("\n\n---------------------\nEverything was written to messages.log.\n--------------------\n\n")

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
