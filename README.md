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


