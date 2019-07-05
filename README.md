# Coding-Samples
Coding-Samples

30/06/2019

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
