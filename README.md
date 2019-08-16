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


Creating Sliders
# Perform necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider

# Create first slider: slider1
slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)

# Create second slider: slider2
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)

# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1, slider2)

# Add the layout to the current document
curdoc().add_root(layout)



GRAPH WITH DROP DOWN MENU

# Perform necessary imports
from bokeh.models import ColumnDataSource, Select

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)

# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }

# Create a dropdown Select widget: select    
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)




MAKING A BUTTON
# Create a Button with label 'Update Data'
button = Button(label='Update Data')

# Define an update callback with no arguments: update
def update():

    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)

    # Update the ColumnDataSource data dictionary
    source.data = {'x': x, 'y': y}

# Add the update callback to the button
button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)

# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)


MAKING BEAUTIFUL INTERACTIVE GAP MINDER DATA PLOT
# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)

BERNOULLI TRIAL
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success
    
    
Linear Regression using polyfit()

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, deg=1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0, 100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()
    
    

Statisticall Analysing Darwins Finch Data
Initial look

# Create bee swarm plot
_ = sns.swarmplot(x = 'year', y = 'beak_depth', data =df)
# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')
# Show the plot
plt.show()

Taking a look using ECDFs
# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)
# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')
# Set margins
plt.margins(0.02)
# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')
# Show the plot
plt.show()



