import os
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from id3 import Id3Estimator
from id3 import export_graphviz

# This class handles the data transformation, cleaning, tree generation and scoring
class DataPredictionMachine:
    
    #initialize with data
    def __init__(self, run_id, data, print_commands):
        self.dataframe = data
        self.run_id = run_id
        self.estimator = None 
        self.print_commands = print_commands

    # return the dataframe
    def get_dataframe(self):
        return self.dataframe

    # -----------------------------------------------------
    # DATA UNDERSTANDING
    # -----------------------------------------------------
    
    # DATA DESCRIPTION
    # describe dataset
    def describe(self):
        self.__print("\n-------------------------------------- DATA UNDERSTANDING ------------------------------------------\n")
        self.__print("Data preview:\n")
        self.__print(self.dataframe.head())
        self.__print("\n")
        self.__print("Data shape:\n")
        self.__print(self.dataframe.shape)
        self.__print("\n")
        self.__print("Data description:\n")
        self.__print(self.dataframe.describe())
        self.__print("\n")
        self.__print("Categorical Attributes:\n")
        self.__print("All titles: ")
        self.__print(self.__getAllTitles())
        self.__print("All decks:")
        self.__print(self.__getAllDecks())
        self.__print("All ticket sequences:")
        self.__print(self.__getAllTicketNrs())
        self.__print("All embarked:")
        self.__print(self.__getAllPorts())
        self.__print("Visualize Missing Data:\n")
        self.__visualizeMissings()
    
    # create missing values matrix
    def __visualizeMissings(self):
        self.__create_output_dir()
        file_path = "./output/" + "/dataset_missing-values-matrix.png"
        msno.matrix(self.dataframe).figure.savefig(file_path)
        #plt.show()
        #plt.clf()
    
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
    
    def __extractDeck(self, cabin):
        try:
            res = cabin[0]
        except:
            res = np.NaN
        return res
    
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
    
    # DESCRIBING IMAGES
    # describe the dataset as images and save them
    def create_describing_images(self):
        self.__create_output_dir()
        if self.run_id and not os.path.exists("./output/plots"):
            os.makedirs("./output/plots")
        
        self.__draw_single_plots()
        self.__draw_swarmplot_and_lmplots()
        
        file_path = "./output/plots" + "/pairGrid.png"
        g = sns.PairGrid(self.dataframe, hue="Survived")
        g.map_diag(plt.hist)
        g.map_upper(sns.regplot)
        g.map_lower(sns.violinplot)
        g.add_legend()
        g.savefig(file_path)
        plt.show()
        plt.clf()
    
    def __draw_single_plots(self):
        for col in ["Survived", "Sex", "Pclass", "SibSp", "Parch", "Embarked", "Familysize"]:
            file_path = "./output/plots" + "/" + col + ".png"
            sns.factorplot(col, data=self.dataframe, kind="count").savefig(file_path)
            plt.show()
            plt.clf()
        for col in ["Age", "Fare", "FarePerPerson"]:
            file_path = "./output/plots" + "/histograms_" + col + ".png"
            fg = sns.FacetGrid(data=self.dataframe)
            fg.map(sns.kdeplot, col, shade=True)
            fg.savefig(file_path)
            plt.show()
            plt.clf()
    
    def __draw_swarmplot_and_lmplots(self):
        for var1 in ["Survived", "Sex", "Pclass", "SibSp", "Parch", "Embarked", "Familysize", "Age", "Fare", "FarePerPerson"]:
            for var2 in ["Survived", "Sex", "Pclass", "SibSp", "Parch", "Embarked", "Familysize", "Age", "Fare", "FarePerPerson"]:
                if(not(var1 == var2)):
                    if(not((self.dataframe[var1].dtypes == np.float64 or self.dataframe[var1].dtypes == np.int64) or (self.dataframe[var2].dtypes == np.float64 or self.dataframe[var2].dtypes == np.int64))):
                        print("not numeric")
                        print(self.dataframe[var1].dtypes)
                        print(self.dataframe[var2].dtypes)
                    else:
                        file_path = "./output/plots" + "/swarmplot_" + var1 + "-" + var2 + ".png"
                        sns.swarmplot(x=var1, y=var2, hue="Survived", data=self.dataframe).figure.savefig(file_path)
                        plt.show()
                        plt.clf()
                    
                    if((self.dataframe[var1].dtypes == np.float64 or self.dataframe[var1].dtypes == np.int64) and (self.dataframe[var2].dtypes == np.float64 or self.dataframe[var2].dtypes == np.int64)):
                        file_path = "./output/plots" + "/lmplot_" + var1 + "-" + var2 + ".png"
                        sns.lmplot(x=var1, y=var2, data=self.dataframe, y_jitter=.03).savefig(file_path)
                        plt.show()
                        plt.clf()
                        file_path = "./output/plots" + "/lmplot_marked_" + var1 + "-" + var2 + ".png"
                        sns.lmplot(x=var1, y=var2, hue="Survived", data=self.dataframe, y_jitter=.03).savefig(file_path)
                        plt.show()
                        plt.clf()

    # -----------------------------------------------------
    # DATA PREPARATION
    # -----------------------------------------------------
    # prepare dataset
    def prepare(self):
        self.__print("\n-------------------------------------- DATA PREPARATION ------------------------------------------\n")
        self.__print("Extract Combined Attributes\n")
        self.__pepare_titles()
        self.__prepare_surnames()
        self.__prepare_firstnames()
        self.__prepare_brakets()
        self.__prepare_decks()
        self.__prepareNrOfCabins()
        self.__prepareFamilysize()
        self.__calcFaresPerCabin()
        self.__calcFaresPerPerson()
        self.__prepare_ticket_sequence()

    # EXTRACT COMBINED ATTRIBUTES
    def __pepare_titles(self):
        # create new column named "Title" for all titles
        self.dataframe['Title'] = self.dataframe['Name'].map(lambda name: self.__extractTitle(name))
    
    def __splitBrakets(self, name):
        try:
            res = name.split('(')[1].split(')')[0].strip()
        except:
            res = np.NaN
        return res
    
    def __prepare_surnames(self):
        self.dataframe['Surname'] = self.dataframe['Name'].map(lambda name:name.split(',')[0].strip())
    
    def __prepare_firstnames(self):
        self.dataframe['FirstNames'] = self.dataframe['Name'].map(lambda name:name.split('.')[1].split('(')[0].strip())
    
    def __prepare_brakets(self):
        self.dataframe['Brakets'] = self.dataframe['Name'].map(lambda name: self.__splitBrakets(name))
    
    def __prepare_decks(self):
        self.dataframe['Deck'] = self.dataframe['Cabin'].map(lambda cabin:self.__extractDeck(cabin))
    
    def __prepareNrOfCabins(self):
        self.dataframe['NrOfCabins'] = self.dataframe['Cabin'].map(lambda cabin:self.__calcNrOfCabins(cabin))
    
    def __calcNrOfCabins(self, cabin):
        cabinCounter = 0
        try:
            cabins = cabin.split(" ")
            for c in cabins:
                if(len(c)>2):
                    cabinCounter += 1
        except:
            cabinCounter = np.NaN
        return cabinCounter
    
    def __calcFaresPerCabin(self):
        row = 0
        self.dataframe['FarePerCabin'] = np.nan
        for cabinCounter in self.dataframe['NrOfCabins']:
            temp_fare = (self.dataframe.loc[row, 'Fare'])
            if(np.isnan(cabinCounter)):
                self.dataframe.loc[row, 'FarePerCabin'] = np.NaN
            else:
                if cabinCounter>1:
                    #Preis durch die Anzahl der Cabinen
                    self.dataframe.loc[row, 'FarePerCabin'] = temp_fare / cabinCounter
                else:
                    self.dataframe.loc[row, 'FarePerCabin'] = temp_fare
            row = row + 1
    
    def __calcFaresPerPerson(self):
        row = 0
        self.dataframe['FarePerPerson'] = np.nan
        for familysize in self.dataframe['Familysize']:
            temp_fare = (self.dataframe.loc[row, 'Fare'])
            if(np.isnan(familysize)):
                self.dataframe.loc[row, 'FarePerPerson'] = np.NaN
            else:
                if familysize>1:
                    #Preis durch die Anzahl der Cabinen
                    self.dataframe.loc[row, 'FarePerPerson'] = temp_fare / familysize
                else:
                    self.dataframe.loc[row, 'FarePerPerson'] = temp_fare
            row = row + 1
    
    def __prepareFamilysize(self):
        row = 0
        self.dataframe['Familysize'] = np.nan
        for parch,sibsp in zip(self.dataframe['Parch'], self.dataframe['SibSp']):
            self.dataframe.loc[row, 'Familysize'] = 1 + parch + sibsp
            row = row + 1
        #self.dataframe['Familysize'] = self.dataframe['Parch', 'Sibsp'].map(lambda (parch,sibsp):self.__calcFamilysize(parch, sibsp))
    
    def __calcFamilysize(self, parch, sibsp):
        return (1 + parch + sibsp)
    
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
    
    # HANDLE MISSING VALUES
    # handle all missing values of the dataset
    def handle_missing_values(self):
        print("\n--------------------------------------Handle Missing Values------------------------------------------\n")
        self.__create_output_dir()
        file_path = "./output/" + "/pre-handling_missing-values-matrix.png"
        msno.matrix(self.dataframe).figure.savefig(file_path)
        #plt.show()
        #plt.clf()
        self.__findAllMissingValueTypes()
        self.__create_output_dir()
        file_path = "./output/" + "/post-handling_missing-values-matrix.png"
        msno.matrix(self.dataframe).figure.savefig(file_path)
        #plt.show()
        #plt.clf()
    
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
    
    # CREATE DUMMY VARS
    # creating dummy vars for all categorical attributes
    def create_dummy_vars(self):
        self.__print("Creating Dummy Vars: \n")
        self.__create_dummy_vars_from_title()
        self.__create_dummy_vars_from_sex()
        self.__create_dummy_vars_from_pclass()
        self.__create_dummy_vars_from_embarked()
    
    def __create_dummy_vars_from_sex(self):
        self.__print("create_dummy_vars_from_sex():  Sex -> Male / Female \n")
        self.dataframe['Male'] = self.dataframe['Sex'].map(lambda x: x == "male")
        self.dataframe['Female'] = self.dataframe['Sex'].map(lambda x: x == "female")
    
    def __create_dummy_vars_from_pclass(self):
        self.__print("create_dummy_vars_from_pclass():  Pclass -> FirstClass / SecondClass / ThirdClass \n")
        self.dataframe['FirstClass'] = self.dataframe['Pclass'].map(lambda x: x == 1)
        self.dataframe['SecondClass'] = self.dataframe['Pclass'].map(lambda x: x == 2)
        self.dataframe['ThirdClass'] = self.dataframe['Pclass'].map(lambda x: x == 3)
        
    def __create_dummy_vars_from_embarked(self):
        self.__print("create_dummy_vars_from_embarked():  Embarked -> PortS / PortC / PortQ \n")
        self.dataframe['PortS'] = self.dataframe['Embarked'].map(lambda x: x == 'S')
        self.dataframe['PortC'] = self.dataframe['Embarked'].map(lambda x: x == 'C')
        self.dataframe['PortQ'] = self.dataframe['Embarked'].map(lambda x: x == 'Q')
    
    def __create_dummy_vars_from_title(self):
        self.__print("create_dummy_vars_from_title():  Title -> Royal / Rare / Child \n")
        self.dataframe['Royal'] = self.dataframe['Title'].map(lambda title: title == "Royal")
        self.dataframe['Rare'] = self.dataframe['Title'].map(lambda title: title == "Rare")
        self.dataframe['Child'] = self.dataframe['Title'].map(lambda title: title == "Child")
    
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
    
    def __prepare_ticket_sequence(self):
        self.dataframe['PreTicketSequence'] = self.dataframe['Ticket'].map(lambda ticketnr: self.__extractTicketNr(ticketnr))
    
    # FINAL CLEAN
    # delete unused columns that are not needet for the model
    def clean(self):
        self.__print("\n--------------------------------------DATASET CLEANING------------------------------------------\n")
        #columns = ['Pclass', 'Sex', 'Deck', 'Name', 'PassengerId', "Title", "Ticket", "Name", "Surname", "FirstNames", "Brakets", "Cabin", "Embarked", "PreTicketSequence", "Fare"]
        columns = ['Pclass', 'Sex', 'Name', 'PassengerId', "Title", "Ticket", "Name", "Surname", "FirstNames", "Embarked"]
        self.__print("Deleting columns:")
        self.__print(columns)
        self.dataframe.drop(columns, inplace=True, axis=1)

    # print a short preview of the dataset
    def preview(self):
        self.__print("\n--------------------------------------DATASET PREVIEW------------------------------------------\n")
        self.__print(self.dataframe.head())
    
    # -----------------------------------------------------
    # MODELLING
    # -----------------------------------------------------
    
    # generate a ID3 tree
    def generate_tree(self, max_depth):
        self.__print("\n-------------------------------------- Modelling ------------------------------------------\n")
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
        dot_data = export_graphviz(self.estimator.tree_, './output/' + self.run_id + '/tree.dot', feature_names)
        # create png file
        #command = ["dot", "-Tpng", './output/' + self.run_id + '/tree.dot', "-o", "./output/" + self.run_id + "/tree.png"]
        #subprocess.check_call(command, shell=True)
        command = "dot -Tpng " + './output/' + self.run_id + '/tree.dot' + " -o "+ "./output/" + self.run_id + "/tree.png" #Tsvg can be changed to Tjpg, Tpng, Tgif etc (see dot man pages)
        os.system(command)
    
    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------
    
    # predict the survival for the given dataframe
    def predict(self, other_dataframe):
        return self.estimator.predict(other_dataframe)
    
    def evaluate(self, other_dataframe):
        self.__print("\n-------------------------------------- EVALUATION ------------------------------------------\n")
        actual_survival_array = other_dataframe['Survived'].values
        predictet_survival_array = self.predict(other_dataframe.iloc[:, 1:])
        from sklearn.metrics import confusion_matrix
        confus_matrix = confusion_matrix(actual_survival_array,predictet_survival_array)
        r1 = confus_matrix [0]
        r2 = confus_matrix [1]
        
        tp = r1[0]
        fp = r1[1]
        fn = r2[0]
        tn = r2[1]
        
        cp = tp+fn
        cn = fp+tn
        pcp = tp + fp
        pcn = fn + tn
        
        
        true_positive_rate = tp / (cp)
        false_negative_rate = fn / (cp)
        true_negative_rate = tn / (cn)
        false_positive_rate = fp / (cn)
        
        positive_prediction_value  = tp / (pcp)
        false_omission_rate  = fn / (pcn)
        false_discovery_rate  = fp / (pcp)
        negative_prediction_value  = tn / (pcn)
        
        f_measure = 2 * ((positive_prediction_value*true_positive_rate)/(positive_prediction_value+true_positive_rate))
        accuracy = (tp + tn)/(cp + cn)
        
        
        self.__print("true_positive_rate: "+str(true_positive_rate))
        self.__print("false_negative_rate: "+str(false_negative_rate))
        self.__print("true_negative_rate: "+str(true_negative_rate))
        self.__print("false_positive_rate: "+str(false_positive_rate))
        self.__print("positive_prediction_value: "+str(positive_prediction_value))
        self.__print("false_omission_rate: "+str(false_omission_rate))
        self.__print("false_discovery_rate: "+str(false_discovery_rate))
        self.__print("negative_prediction_value: "+str(negative_prediction_value))
        self.__print("f_measure: "+str(f_measure))
        self.__print("accuracy: "+str(accuracy))
    
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
