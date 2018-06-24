import os
import missingno as msno
import numpy as np
import subprocess
from id3 import Id3Estimator
from id3 import export_graphviz

# This class handles the data transformation, cleaning, tree generation and scoring
class DataPredictionMachine:
    # dataframe (Pandas obj)
    # run_id (String) = output folder names for all files
    # print_commands (Bool) = log to console
    def __init__(self, run_id, dataframe, print_commands):
        self.dataframe = dataframe
        self.run_id = run_id
        self.estimator = None 
        self.print_commands = print_commands

    # return the dataframe
    def get_dataframe(self):
        return self.dataframe

    # describe dataset
    def describe(self):
        self.__print("\n--------------------------------------DATA DESCRIPTION------------------------------------------\n")
        self.__print("Data preview:\n")
        self.__print(self.dataframe.head())
        self.__print("\n")
        self.__print("Data shape:\n")
        self.__print(self.dataframe.shape)
        self.__print("\n")
        self.__print("Data description:\n")
        self.__print(self.dataframe.describe())

    # transform the data
    def prepare(self):
        self.__print("\n--------------------------------------PREPARATION START------------------------------------------\n")
        self.__print("All titles: ")
        self.__print(self.__getAllTitles())
        self.__pepare_titles()
        self.__create_dummy_vars_from_title()
        self.__prepare_surnames()
        self.__prepare_firstnames()
        self.__prepare_brakets()
        self.__print("All decks:")
        self.__print(self.__getAllDecks())
        self.__prepare_decks()
        self.__create_dummy_vars_from_decks()
        self.__faresCleaning()
        self.__print("All ticket sequences:")
        self.__print(self.__getAllTicketNrs())
        self.__prepare_ticket_sequence()
        self.__create_dummy_vars_from_sex()
        self.__create_dummy_vars_from_pclass()
        self.__print("All embarked:")
        self.__print(self.__getAllPorts())
        self.__create_dummy_vars_from_embarked()

    # describe the dataset as images and save them
    def create_describing_images(self):
        self.__create_output_dir()
        file_path = "./output/" + self.run_id + "/missing-values-matrix.png"
        msno.matrix(self.dataframe).figure.savefig(file_path)

    # handle all missing values of the dataset
    def handle_missing_values(self):
        print("\n--------------------------------------Handle Missing Values------------------------------------------\n")
        
        print(self.dataframe.isnull().sum())
        self.__findAllMissingValueTypes()
        print(self.dataframe.isnull().sum())

        # TODO: DELETE! THIS IS JUST FOR MAKING THE MODEL WORK (DOES NOT WORK WITH STRING ATTRIBUTES CURRENTLY)
        #self.dataframe = self.dataframe.fillna(0)
    
    def __findAllMissingValueTypes(self):
        for column,numberOfMissings in (self.dataframe.isnull().sum().items()):
            if(numberOfMissings>0):
                if(numberOfMissings > self.dataframe.shape[0]*2//3):
                    print("'" + column.upper() + "' has to many missing values")
                    self.dataframe.drop(column, axis = 1, inplace = True)
                    print("Dropped '" + column.upper() + "'")
                else:
                    self.__findMissingValueType(column)
    
    def __findMissingValueType(self, column):
        self.__createBinaryMissingClassification(column)
        name = column + 'Missing'
        corr = self.dataframe.corr()
        np.fill_diagonal(corr.values, 0)
        corr_filtered = corr.filter([name], axis=0)
        corr_triu = corr_filtered.where(~np.tril(np.ones(corr_filtered.shape)).astype(np.bool))
        corr_triu = corr_triu.stack()
        corr_triu = corr_triu[(corr_triu > 0.7) | (corr_triu < -0.7)]
        if(corr_triu.size == 0):
            print("'" + column.upper() + "' is Missing Completly At Random")
            if(self.dataframe[column].dtypes == np.float64 or self.dataframe[column].dtypes == np.int64):
                self.dataframe[column].fillna(self.dataframe[column].mean(), inplace=True)
                print("Filled missing values with mean for '" + column.upper() + "'")
            else:
                self.dataframe[column].fillna(self.dataframe[column].mode()[0], inplace=True)
                print("Filled missing values with mode for '" + column.upper() + "'")
        else:
            print("'" + column.upper() + "' is Missing At Random or Missing Not At Random")
            self.dataframe.drop(column, axis = 1, inplace = True)
            print("Dropped '" + column.upper() + "'")
        self.dataframe.drop(name, axis = 1, inplace = True)
    
    def __createBinaryMissingClassification(self, column):
        name = column + 'Missing'
        self.dataframe[name] = self.dataframe[column].map(lambda x:self.__getMissing(x))
    
    def __getMissing(self, x):
        if(x != x):
            return False
        else:
            return True

    # delete unused columns that are not needet for the model
    def clean(self):
        self.__print("\n--------------------------------------DATASET CLEANING------------------------------------------\n")
        #columns = ['Pclass', 'Sex', 'Deck', 'Name', 'PassengerId', "Title", "Ticket", "Name", "Surname", "FirstNames", "Brakets", "Cabin", "Embarked", "PreTicketSequence", "Fare"]
        columns = ['Pclass', 'Sex', 'Name', 'PassengerId', "Title", "Ticket", "Name", "Surname", "FirstNames", "Embarked", "Fare"]
        self.__print("Deleting columns:")
        self.__print(columns)
        self.dataframe.drop(columns, inplace=True, axis=1)

    # print a short preview of the dataset
    def preview(self):
        self.__print("\n--------------------------------------DATASET PREVIEW------------------------------------------\n")
        self.__print(self.dataframe)
        # TODO: add image generation

    # generate a ID3 tree
    def generate_tree(self, max_depth):
        self.__create_output_dir()
        self.__print("\n--------------------------------------DECISION TREE GENERATION------------------------------------------\n")
        self.__print("Output file names: ./output/" + self.run_id + "/tree.dot ./output/" + self.run_id + "/tree.png")
        # the estimator
        self.estimator = Id3Estimator(max_depth)
        # suvrived
        x = self.dataframe.iloc[:, 0]
        # all attributes except survieved
        y = self.dataframe.iloc[:, 1:]
        # all var names except survieved
        feature_names = list(y.columns.values)
        # calc the tree
        self.estimator = self.estimator.fit(y, x)
        # export as .dot
        tree = export_graphviz(self.estimator.tree_, "./output/" + self.run_id + '/tree.dot', feature_names)
        # create png file
        command = ["dot", "-Tpng", "./output/" + self.run_id + "/tree.dot", "-o", "./output/" + self.run_id + "/tree.png"]
        subprocess.check_call(command)

    # predict the survival for the given dataframe
    def predict(self, other_dataframe):
        return self.estimator.predict(other_dataframe)

    # score the current model with a given dataframe
    # this dataframe needs to have the survival columns as the first column
    def calc_score(self, other_dataframe):
        self.__print("\n--------------------------------------SCORING------------------------------------------\n")
        actual_survival_array = other_dataframe['Survived'].values
        predictet_survival_array = self.predict(other_dataframe.iloc[:, 1:])
        # self.__print("predictet_survival_array:\n", predictet_survival_array, "\n")

        right_predicted = 0
        total_array_len = len(actual_survival_array)
        # break if something is wrong with the input data..
        if(total_array_len != len(predictet_survival_array)):
            return
        # count how many values are perfect predicted
        for i in range(total_array_len):
            if(actual_survival_array[i] == predictet_survival_array[i]):
                right_predicted += 1
        # return percentage
        return right_predicted/total_array_len

    # ----------------------------------------------------------------------------------------------------
    # HELPER METHODS
    # ----------------------------------------------------------------------------------------------------
    # just print if it is wanted
    def __print(self, args):
        if(self.print_commands):
            print(args)

    # creates a folder for the given run_id
    # call this method in every method that outputs a file
    # so all methods stay independent
    def __create_output_dir(self):
        if self.run_id and not os.path.exists("./output/" + self.run_id):
            os.makedirs("./output/" + self.run_id)

    # -----------------------------------------------------------
    # TITLES
    # -----------------------------------------------------------
    # returns all titles as a dict
    # {'Mr': 517, 'Mrs': 126, 'Miss': 185, 'Master': 40, 'Don': 1, 'Rev': 6, 'Dr': 7, 'Major': 2, 'Lady': 1, 'Sir': 1, 'Col': 2, 'Capt': 1, 'the Countess': 1, 'Jonkheer': 1}
    def __getAllTitles(self):
        titles = {}
        for name in self.dataframe['Name']:
            title = self.__extractTitle(name)
            if (title in titles):
                titles[title] += 1
            else:
                titles[title] = 1
                return titles

    # categorize the title
    def __extractTitle(self, name):
        # get the title
        title = name.split(',')[1].split('.')[0].strip()
        # group all rare titles
        rare_titles = ["Lady", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Jonkheer", "Dona"]
        for t in rare_titles:
            title.replace(t, "Rare")

        # 'Countess', 'Lady', 'Sir' -> 'Royal'
        royal_titles = ["Countess", "Lady", "Sir"]
        for t in royal_titles:
            title.replace(t, "Royal")
        # group all misses
        title = title.replace("Mlle", "Miss")
        title = title.replace("Ms", "Miss")
        # all mrs
        title = title.replace("Mme", "Mrs")
        # all people who have masters are children unter 12
        title = title.replace("Master", "Child")
        return title

    def __pepare_titles(self):
        # create new column named "Title" for all titles
        self.dataframe['Title'] = self.dataframe['Name'].map(lambda name: self.__extractTitle(name))

    def __create_dummy_vars_from_title(self):
        self.__print("create_dummy_vars_from_title():  Title -> Royal / Rare / Child \n")
        self.dataframe['Royal'] = self.dataframe['Title'].map(lambda title: title == "Royal")
        self.dataframe['Rare'] = self.dataframe['Title'].map(lambda title: title == "Rare")
        self.dataframe['Child'] = self.dataframe['Title'].map(lambda title: title == "Child")

    # -----------------------------------------------------------
    # SURNAME (We do nut use this..)
    # -----------------------------------------------------------
    # extract brakets from name
    def __splitBrakets(self, name):
        try:
            res = name.split('(')[1].split(')')[0].strip()
        except:
            res = np.NaN
            return res

    # extract surname
    def __prepare_surnames(self):
        self.dataframe['Surname'] = self.dataframe['Name'].map(lambda name:name.split(',')[0].strip())

    # extract name
    def __prepare_firstnames(self):
        self.dataframe['FirstNames'] = self.dataframe['Name'].map(lambda name:name.split('.')[1].split('(')[0].strip())

    # extract content in brakets
    def __prepare_brakets(self):
        self.dataframe['Brakets'] = self.dataframe['Name'].map(lambda name: self.__splitBrakets(name))

    # -----------------------------------------------------------
    # CABINS
    # -----------------------------------------------------------
    def __extractDeck(self, cabin):
        try:
            res = cabin[0]
        except:
            res = np.NaN
            return res

    # extract deck
    def __prepare_decks(self):
        self.dataframe['Deck'] = self.dataframe['Cabin'].map(lambda cabin:self.__extractDeck(cabin))

    # returns all decks from the dataset as a dict
    # {nan: 687, 'C': 59, 'E': 32, 'G': 4, 'D': 33, 'A': 15, 'B': 47, 'F': 13, 'T': 1}
    def __getAllDecks(self):
        decks = {}
        for cabin in self.dataframe['Cabin']:
            deck = self.__extractDeck(cabin)
            if (deck in decks):
                decks[deck] += 1
            else:
                decks[deck] = 1
                return decks

    def __create_dummy_vars_from_decks(self):
        self.__print("create_dummy_vars_from_decks():  Deck -> DeckA / DeckB / DeckC / DeckD / DeckE / DeckF / DeckG / DeckT \n")
        self.dataframe['DeckA'] = self.dataframe['Deck'].map(lambda x: x == "A")
        self.dataframe['DeckB'] = self.dataframe['Deck'].map(lambda x: x == "B")
        self.dataframe['DeckC'] = self.dataframe['Deck'].map(lambda x: x == "C")
        self.dataframe['DeckD'] = self.dataframe['Deck'].map(lambda x: x == "D")
        self.dataframe['DeckE'] = self.dataframe['Deck'].map(lambda x: x == "E")
        self.dataframe['DeckF'] = self.dataframe['Deck'].map(lambda x: x == "F")
        self.dataframe['DeckG'] = self.dataframe['Deck'].map(lambda x: x == "G")
        self.dataframe['DeckT'] = self.dataframe['Deck'].map(lambda x: x == "T")

    # -----------------------------------------------------------
    # FARES
    # -----------------------------------------------------------
    def __faresCleaning(self):
        row = 0
        self.dataframe['CleanedFares'] = np.nan
        for cabin in self.dataframe['Cabin']:
            temp_fare = (self.dataframe.loc[row, 'Fare'])
            if (cabin is not np.nan):
                cabinNumberTotal = self.__countCabineNr(cabin)
                if cabinNumberTotal>1:
                    #Preis durch die Anzahl der Cabinen
                    self.dataframe.loc[row, 'CleanedFares'] = temp_fare / cabinNumberTotal
                else:
                    self.dataframe.loc[row, 'CleanedFares'] = temp_fare
            else:
                self.dataframe.loc[row, 'CleanedFares'] = temp_fare
            row = row + 1

    def __countCabineNr(self, cabinenNr):
        cabinCounter = 0
        #Mehr als 9 Zimmer sind eh nie in einer Zeile, daher hier max 10 Splits
        cabinenNrSplitted = cabinenNr.split(' ',10)
        # was nicht abgefangen wird, ist ob nur der Deckbereich drin steht,
        # aber das kommt nicht in unseren Daten vor, daher hier vernachlässigt
        if (len(cabinenNrSplitted[0])==1):
            i = 1
            while i < (len(cabinenNrSplitted)):
                if(int(len(cabinenNrSplitted[i])>0)):
                    cabinCounter = cabinCounter + 1
                    i = i + 1
        elif (len(cabinenNrSplitted[0])>0):
            i = 0
            while i < (len(cabinenNrSplitted)):
                if(int(len(cabinenNrSplitted[i])>0)):
                    cabinCounter = cabinCounter + 1
                    i = i + 1
        else:
            cabinCounter = 1
        return cabinCounter

    # -----------------------------------------------------------
    # TICKETS (We do nut use this..)
    # -----------------------------------------------------------
    # returns all ticket numbers from the dataset
    def __getAllTicketNrs(self):
        ticketnrs = {}
        for ticketnr in self.dataframe['Ticket']:
            preSequence = self.__extractTicketNr(ticketnr)
            if (preSequence in ticketnrs):
                ticketnrs[preSequence] += 1
            else:
                ticketnrs[preSequence] = 1
                return ticketnrs

    # extract tichet nr
    def __extractTicketNr(self, ticketnr):
        try:
            if(any(c.isalpha() for c in ticketnr)):
                res = ticketnr.split(' ')[0].strip()
            else:
                res = np.NaN
        except:
            res = np.NaN
        return res

    # extract pre sequence
    def __prepare_ticket_sequence(self):
        self.dataframe['PreTicketSequence'] = self.dataframe['Ticket'].map(lambda ticketnr: self.__extractTicketNr(ticketnr))

    # -----------------------------------------------------------
    # SEX
    # -----------------------------------------------------------
    def __create_dummy_vars_from_sex(self):
        self.__print("create_dummy_vars_from_sex():  Sex -> Male / Female \n")
        self.dataframe['Male'] = self.dataframe['Sex'].map(lambda x: x == "male")
        self.dataframe['Female'] = self.dataframe['Sex'].map(lambda x: x == "female")

    # -----------------------------------------------------------
    # CLASSES
    # -----------------------------------------------------------
    def __create_dummy_vars_from_pclass(self):
        self.__print("create_dummy_vars_from_pclass():  Pclass -> FirstClass / SecondClass / ThirdClass \n")
        self.dataframe['FirstClass'] = self.dataframe['Pclass'].map(lambda x: x == 1)
        self.dataframe['SecondClass'] = self.dataframe['Pclass'].map(lambda x: x == 2)
        self.dataframe['ThirdClass'] = self.dataframe['Pclass'].map(lambda x: x == 3)

    # -----------------------------------------------------------
    # PORTS
    # -----------------------------------------------------------
    # returns all ports from the dataset
    # {'S': 644, 'C': 168, 'Q': 77, nan: 2} 
    def __getAllPorts(self):
        ports = {}
        for port in self.dataframe['Embarked']:
            if (port in ports):
                ports[port] += 1
            else:
                ports[port] = 1
                return ports

    def __create_dummy_vars_from_embarked(self):
        self.__print("create_dummy_vars_from_embarked():  Embarked -> PortS / PortC / PortQ \n")
        self.dataframe['PortS'] = self.dataframe['Embarked'].map(lambda x: x == 'S')
        self.dataframe['PortC'] = self.dataframe['Embarked'].map(lambda x: x == 'C')
        self.dataframe['PortQ'] = self.dataframe['Embarked'].map(lambda x: x == 'Q')