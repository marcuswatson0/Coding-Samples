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

