import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pulp import *
from EncodeNames import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Import Data
raw_data = get_data('C:/Users/victo/OneDrive/Desktop/Group Gen/Test1.csv')
raw_data_labels = raw_data.columns


# combine first and last names
df = raw_data.copy()
df["Full_Name"] = df["First Name "] + ' ' + df["Last Name "]

# preprocess and clean data
labels = ["Full_Name", "Gender ", "Badminton Skill Level ", "Which Club are you a member of?"]
preprocessed_data = clean_data(df, labels)
num_advance = count_exact_matches(preprocessed_data,"Advanced (Very confident in skills, likely participated in tournaments or leagues before)")
num_inter = count_exact_matches(preprocessed_data,"Intermediate ( Solid grasp of techniques , got some experience under your belt)")
num_begin = count_exact_matches(preprocessed_data,"Beginner ( This is your first time playing badminton, or you don't have much experience)")


# Encode the data
skill = ["Beginner ( This is your first time playing badminton, or you don't have much experience)", "Intermediate ( Solid grasp of techniques , got some experience under your belt)", "Advanced (Very confident in skills, likely participated in tournaments or leagues before)"]
gender = ["Male", "Female", "Prefer not to say"]
club = ["ESA", "UABC", "Non-Member"]
encoded_data = encode_names(preprocessed_data, labels, skill, gender, club)
print(encoded_data.head())



# Transpose data to create Binary Integer Program
encoded_data_lp = encoded_data.transpose(copy=True)
encoded_data_lp = encoded_data_lp.rename(columns=encoded_data_lp.iloc[0])
encoded_data_lp = encoded_data_lp.drop(encoded_data_lp.index[0])
encoded_data_lp = simply_gender(encoded_data_lp,"Gender ") # Assume Non-specified gender would get along with girls
encoded_data_lp = simply_club(encoded_data_lp,"Which Club are you a member of?") # Assume non members are for ESA
print(encoded_data_lp)





# Groups Dataframe

groups ={}
group = []

# Create BIP dictionary
people = {}
cost = {}
for i in encoded_data["Full_Name"]:
    people.update({i:0})
    cost.update({i:1})

print(cost)


# BIP
prob = LpProblem("UABC_Social_Event_Problem", LpMaximize)
variables = LpVariable.dicts("People", people,0,1,LpInteger)


# Minimise
prob += lpSum([cost[index] * variables[index] for index in variables])

# Subject to:
prob += lpSum([variables[index] for index in variables]) <= 10 # Number of people per group
#prob += lpSum([variables[index] for index in variables]) >= 10

prob += lpSum([encoded_data_lp.loc["Gender ", index] * variables[index] for index in variables]) >= 2 # More than 2 girls per group

prob += lpSum([encoded_data_lp.loc["Badminton Skill Level ", index] * variables[index] for index in variables]) <= 18 # Limit number of advance players per group

prob += lpSum([encoded_data_lp.loc["Which Club are you a member of?", index] * variables[index] for index in variables]) <= 14 # Even number of club members per group
for j in range(1,7):
    # Solve problem
    prob.writeLP('BIP_Social_Event_Problem.lp')
    prob.solve()

    for v in prob.variables():
        if v.varValue == 1:
            print(v.name)
            group.append(v.name)
            prob += v == 0
    groups.update({j:group})
    group = []

df = pd.DataFrame.from_dict(groups, orient='index')
df = df.transpose()
df = remove_substring_from_df(df,"People_")
df = replace_substring_in_df(df,"_", " ")
print(df)
df.to_csv('output.csv', index=True)
print(num_advance/6, num_inter/6, num_begin/6)
