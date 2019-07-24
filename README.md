# Coding-Samples 

#Loop over Numpy array
#For loop over np_baseball

for word in np.nditer(np_baseball):
    print(word)
    
#Loop over Pandas DataFrame 

#Use .apply(str.upper) vs iterrows() 
#adding a column
for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len(row["country"])

#this method is not very efficient
for lab, row in cars.iterrows() :
    cars.loc[lab, "COUNTRY"] = row["country"].upper()
print(cars)

#far more concise and efficient
cars['COUNTRY'] = cars['country'].apply(str.upper)
print(cars)

Simulating games of random chance

import numpy as np
np.random.seed(123)
#simulate dice roll 1, 2, 3, 4, 5, 6
dice = np.random.randint(1, 7)
print(dice)
# Use randint() again to see if there is a different result
dice = np.random.randint(1, 7)
print(dice)

Output = 3
Output = 6

Roll the dice. Use randint() to create the variable dice.
Finish the if-elif-else construct by replacing ___:
#If dice is 1 or 2, you go one step down.
#if dice is 3, 4 or 5, you go one step up.
#Else, you throw the dice again. The number of eyes is the number of steps you go up.
#Print out dice and step. Given the value of dice, was step updated correctly?

#Starting step
step = 50
#Roll the dice
dice = np.random.randint(1, 7)
#Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice < 2 and dice > 6 :
    step = step + 1
else :
    step = step + np.random.randint(1, 7)
#Print out dice and step
print(dice)
print(step)

# list comprehension format
[[output expression] for iterator variable in iterable]
# list comprehension using conditions format
[output expression for iterator variable in iterable if predicate expression]

# Making Matrices using nested list comprehensions
Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Querying relational databases (SQLite)
#Import packages
from sqlalchemy import create_engine
import pandas as pd

#Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

#Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Album', engine)

#Print head of DataFrame
print(df.head())

#Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()

#Confirm that both methods yield the same result
print(df.equals(df1))

# Ex: Filtering INNER JOIN queries with condition
df = pd.read_sql_query('SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000',engine)

# Opening and reading flat files from the internet

#Import packages
import matplotlib.pyplot as plt
import pandas as pd

#Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

#Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')

#Print the head of the DataFrame
print(df.head())

#Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()

OR

#Import package
import requests

#Specify the url: url
url = "http://www.datacamp.com/teach/documentation"

#Packages the request, send the request and catch the response: r
r = requests.get(url)

#Extract the response: text
text = r.text

#Print the html
print(text)


# Convert temps_f to celsius: temps_c
temps_c = temps_c.columns.str.replace('F', 'C')

# Merge auto and oil: merged
merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')

# Comparing two DataFrames
#Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender == 'F') & (ri.violation == 'Speeding')]

#Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.driver_gender == 'M') & (ri.violation == 'Speeding')]

#Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize=True))

#Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize=True))
