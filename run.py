import pandas as pd
import missingno as msno
import numpy as np

#describe dataset
dataframe = pd.read_csv('train.csv')
print("\n--------------------------------------DATA DESCRIPTION------------------------------------------\n")
print("Data preview:\n")
print(dataframe.head())
print("\n")
print("Data shape:\n")
print(dataframe.shape)
print("\n")
print("Data description:\n")
print(dataframe.describe())
print("\n--------------------------------------PREPARATION START------------------------------------------\n")

# -----------------------------------------------------------
# TITLES
# -----------------------------------------------------------
# handle name
# extract title
def getAllTitles():
    titles = {}
    for name in dataframe['Name']:
        title = extractTitle(name)
        if (title in titles):
            titles[title] += 1
        else:
            titles[title] = 1
    return titles


# {'Mr': 517, 'Mrs': 126, 'Miss': 185, 'Master': 40, 'Don': 1, 'Rev': 6, 'Dr': 7, 'Major': 2, 'Lady': 1, 'Sir': 1, 'Col': 2, 'Capt': 1, 'the Countess': 1, 'Jonkheer': 1}
def extractTitle(name):
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
    # all people who have masters are slaves.. 
    title = title.replace("Master", "Slave")
    return title

def pepare_titles():
    # create new column named "Title" for all titles
    dataframe['Title'] = dataframe['Name'].map(lambda name:extractTitle(name))

def create_dummy_vars_from_title():
    print("create_dummy_vars_from_title():  Title -> Royal / Rare / Slave \n")
    dataframe['Royal'] = dataframe['Title'].map(lambda title: title == "Royal")
    dataframe['Rare'] = dataframe['Title'].map(lambda title: title == "Rare")
    dataframe['Slave'] = dataframe['Title'].map(lambda title: title == "Slave")

print("All titles: ",  getAllTitles(), "\n")
pepare_titles()
create_dummy_vars_from_title()

# -----------------------------------------------------------
# SURNAME
# -----------------------------------------------------------
# extract brakets
def splitBrakets(name):
    try:
        res = name.split('(')[1].split(')')[0].strip()
    except:
        res = np.NaN
    return res

# extract surname
def prepare_surnames():
    dataframe['Surname'] = dataframe['Name'].map(lambda name:name.split(',')[0].strip())

def prepare_firstnames():
    dataframe['FirstNames'] = dataframe['Name'].map(lambda name:name.split('.')[1].split('(')[0].strip())

def prepare_brakets():
    dataframe['Brakets'] = dataframe['Name'].map(lambda name:splitBrakets(name))

prepare_surnames()
prepare_firstnames()
prepare_brakets()

# -----------------------------------------------------------
# CABINS
# -----------------------------------------------------------
def extractDeck(cabin):
    try:
        res = cabin[0]
    except:
        res = np.NaN
    return res

def prepare_decks():
    dataframe['Deck'] = dataframe['Cabin'].map(lambda cabin:extractDeck(cabin))

def getAllDecks():
    decks = {}
    for cabin in dataframe['Cabin']:
        deck = extractDeck(cabin)
        if (deck in decks):
            decks[deck] += 1
        else:
            decks[deck] = 1
    return decks

# {nan: 687, 'C': 59, 'E': 32, 'G': 4, 'D': 33, 'A': 15, 'B': 47, 'F': 13, 'T': 1}
def create_dummy_vars_from_decks():
    print("create_dummy_vars_from_decks():  Deck -> DeckA / DeckB / DeckC / DeckD / DeckE / DeckF / DeckG / DeckT \n")
    dataframe['DeckA'] = dataframe['Deck'].map(lambda x: x == "A")
    dataframe['DeckB'] = dataframe['Deck'].map(lambda x: x == "B")
    dataframe['DeckC'] = dataframe['Deck'].map(lambda x: x == "C")
    dataframe['DeckD'] = dataframe['Deck'].map(lambda x: x == "D")
    dataframe['DeckE'] = dataframe['Deck'].map(lambda x: x == "E")
    dataframe['DeckF'] = dataframe['Deck'].map(lambda x: x == "F")
    dataframe['DeckG'] = dataframe['Deck'].map(lambda x: x == "G")
    dataframe['DeckT'] = dataframe['Deck'].map(lambda x: x == "T")

print("All decks: ", getAllDecks(), "\n")
prepare_decks()
create_dummy_vars_from_decks()

# -----------------------------------------------------------
# FARES
# -----------------------------------------------------------


def cabinenPreisBereinigung():
    row = 0
    for cabin in dataframe['Cabin']:
        if (cabin is not np.nan):
            anzahl = countCabineNr(cabin)
            if anzahl>1:
                #preis durch die Anzahl der Cabinen
                preis = (dataframe.loc[row, 'Fare'])
                dataframe.loc[row, 'Fare'] = preis / anzahl
        row = row + 1

def countCabineNr(cabinenNr):
    cabinCounter = 0
    #Mehr als 9 Zimmer sind eh nie in einer Zeile, daher hier hart 10
    cabinenNrSplitted = cabinenNr.split(' ',10)
	# was nicht abgefangen wird, ist ob nur der Deckbereich drin steht,
	# aber das kommt nicht in unseren Daten vor, daher vernachlässigt
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

cabinenPreisBereinigung()

# -----------------------------------------------------------
# TICKETS
# -----------------------------------------------------------
def getAllTicketNrs():
    ticketnrs = {}
    for ticketnr in dataframe['Ticket']:
        preSequence = extractTicketNr(ticketnr)
        if (preSequence in ticketnrs):
            ticketnrs[preSequence] += 1
        else:
            ticketnrs[preSequence] = 1
    return ticketnrs

# handle ticket number
# extract pre sequence
def extractTicketNr(ticketnr):
    try:
        if(any(c.isalpha() for c in ticketnr)):
            res = ticketnr.split(' ')[0].strip()
        else:
            res = np.NaN
    except:
        res = np.NaN
    return res

def prepare_ticket_sequence():
    dataframe['PreTicketSequence'] = dataframe['Ticket'].map(lambda ticketnr:extractTicketNr(ticketnr))

print("All ticket sequences: ", getAllTicketNrs(), "\n")
prepare_ticket_sequence()

# -----------------------------------------------------------
# SEX
# -----------------------------------------------------------
def create_dummy_vars_from_sex():
    print("create_dummy_vars_from_sex():  Sex -> Male / Female \n")
    dataframe['Male'] = dataframe['Sex'].map(lambda x: x == "male")
    dataframe['Female'] = dataframe['Sex'].map(lambda x: x == "female")

create_dummy_vars_from_sex()

# -----------------------------------------------------------
# CLASSES
# -----------------------------------------------------------
def create_dummy_vars_from_pclass():
    print("create_dummy_vars_from_pclass():  Pclass -> FirstClass / SecondClass / ThirdClass \n")
    dataframe['FirstClass'] = dataframe['Pclass'].map(lambda x: x == 1)
    dataframe['SecondClass'] = dataframe['Pclass'].map(lambda x: x == 2)
    dataframe['ThirdClass'] = dataframe['Pclass'].map(lambda x: x == 3)

create_dummy_vars_from_pclass()

# -----------------------------------------------------------
# PORTS
# -----------------------------------------------------------
# {'S': 644, 'C': 168, 'Q': 77, nan: 2} 
def getAllPorts():
    ports = {}
    for port in dataframe['Embarked']:
        if (port in ports):
            ports[port] += 1
        else:
            ports[port] = 1
    return ports

def create_dummy_vars_from_embarked():
    print("create_dummy_vars_from_embarked():  Embarked -> PortS / PortC / PortQ \n")
    dataframe['PortS'] = dataframe['Embarked'].map(lambda x: x == 'S')
    dataframe['PortC'] = dataframe['Embarked'].map(lambda x: x == 'C')
    dataframe['PortQ'] = dataframe['Embarked'].map(lambda x: x == 'Q')

print("All embarked: ", getAllPorts(), "\n")
create_dummy_vars_from_embarked()

# -----------------------------------------------------------
# MISSING VALS
# -----------------------------------------------------------

#observe missing values
# print("Missing Values: ")
# msno.matrix(dataframe)
# print(dataframe.isnull().sum())
# print("\n\n\n")

print("\n--------------------------------------DATASET CLEANING------------------------------------------\n")
# clean all useless columns
def delete_usless_columns():
    columns = ['Pclass', 'Sex', 'Deck', 'Name', 'PassengerId', "Title", "Ticket", "Name", "Surname", "FirstNames", "Brakets", "Cabin", "Embarked", "PreTicketSequence"]
    print("Deleting columns: ", columns)
    dataframe.drop(columns, inplace=True, axis=1)

delete_usless_columns()

print("\n--------------------------------------DATASET PREVIEW------------------------------------------\n")
# TODO: DELETE! THIS IS JUST FOR MAKING THE MODEL WORK (DOES NOT WORK WITH STRING ATTRIBUTES CURRENTLY)
dataframe = dataframe.fillna(0)
print(dataframe)

# ----------------------------------------
# DECISION TREE
# ----------------------------------------
# brew install graphviz
# pip3 install IPython 
print("\n--------------------------------------DECISION TREE GENERATION------------------------------------------\n")
import subprocess
from id3 import Id3Estimator
from id3 import export_graphviz
# output file names (.dot and .png)
fileName = "tree2"
print("Output file names: " + fileName + ".dot " + fileName + ".png")
# the estimator
estimator = Id3Estimator()
# suvrived
x = dataframe.iloc[:, 0]
# all attributes except survieved
y = dataframe.iloc[:, 1:]
# all var names except survieved
feature_names = list(y.columns.values)
# calc the tree
estimator = estimator.fit(y, x)
# export as .dot
tree = export_graphviz(estimator.tree_, fileName + '.dot', feature_names)
# create png file
command = ["dot", "-Tpng", fileName + ".dot", "-o", fileName + ".png"]
subprocess.check_call(command)
