#!/usr/bin/env python
# coding: utf-8

# # Mid-course Project: Population and World Cup performance

# # Getting the Data

# In[1]:


# Run this cell without changes
import json
import csv



# In[2]:


# Run this cell without changes
world_cup_file = open(r"C:\Users\VishwaParakala\Downloads\worldcup.json", encoding='utf8')
population_file = open(r"C:\Users\VishwaParakala\Downloads\population_csv.csv")


# # 2018 World Cup Data

# In[3]:


# Replace None with appropriate code
world_cup_data = json.load(world_cup_file)

# Close the file now that we're done reading from it
world_cup_file.close()


# In[4]:


# Run this cell without changes

# Check that the overall data structure is a dictionary
assert type(world_cup_data) == dict

# Check that the dictionary has 2 keys, 'name' and 'rounds'
assert list(world_cup_data.keys()) == ['name', 'rounds']


# # Population Data

# In[5]:


# Replace None with appropriate code
population_data = list(csv.DictReader(population_file))

# Close the file now that we're done reading from it
population_file.close()


# In[6]:


#Run this cell without changes

# Check that the overall data structure is a list
assert type(population_data) == list

# Check that the 0th element is a dictionary
# (csv.DictReader interface differs slightly by Python version;
# either a dict or an OrderedDict is fine here)
from collections import OrderedDict
assert type(population_data[0]) == dict or type(population_data[0]) == OrderedDict


# In[7]:


population_data


# # 1. List of Teams in 2018 World Cup

# In[8]:


# Run this cell without changes
world_cup_data.keys()


# In[9]:


world_cup_data["name"]


# # Extracting Rounds

# In[10]:


# Run this cell without changes
rounds = world_cup_data["rounds"]

print("type(rounds):", type(rounds))
print("len(rounds):", len(rounds))
print("type(rounds[3])", type(rounds[3]))
print("rounds[3]:")
rounds[3]


# # Extracting Matches

# In[11]:


# Replace None with appropriate code
matches = []

# "round" is a built-in function in Python so we use "round_" instead
for round_ in rounds:
    # Extract the list of matches for this round
    round_matches = round_["matches"]
    # Add them to the overall list of matches
    matches.extend(round_matches)

matches[0]


# In[12]:


# Run this cell without changes

# There should be 64 matches. If the length is 20, that means
# you have a list of lists instead of a list of dictionaries
assert len(matches) == 64

# Each match in the list should be a dictionary
assert type(matches[0]) == dict


# # Extracting Teams

# In[13]:


# Run this cell without changes
print(matches[0]["team1"])
print(matches[0]["team2"])


# In[14]:


# Replace None with appropriate code
teams_set = set()

for match in matches:
    # Add team1 name value to teams_set
    teams_set.add(match["team1"]["name"])
    # Add team2 name value to teams_set
    teams_set.add(match["team2"]["name"])

teams = sorted(list(teams_set))
print(teams)


# In[15]:


# Run this cell without changes

# teams should be a list, not a set
assert type(teams) == list

# 32 teams competed in the 2018 World Cup
assert len(teams) == 32

# Each element of teams should be a string
# (the name), not a dictionary
assert type(teams[0]) == str


# # 2. Associating Countries with 2018 World Cup Performance

# # Initializing with Wins Set to Zero

# In[16]:


# Replace None with appropriate code

# Create the variable combined_data as described above
combined_data = {}
for team in teams:
    combined_data[team] = {"wins":0}
    combined_data = {team: {"wins": 0} for team in teams}
    

combined_data    


# In[17]:


# Run this cell without changes

# combined_data should be a dictionary
assert type(combined_data) == dict

# the keys should be strings
assert type(list(combined_data.keys())[0]) == str

# the values should be dictionaries
assert combined_data["Japan"] == {"wins": 0}


# # Adding Wins from Matches

# In[18]:


# Replace None with appropriate code

def find_winner(match):
    score_1 = match["score1"]
    score_2 = match["score2"]
    if score_1 > score_2:
        return match["team1"]["name"]
   
    elif score_2> score_1:
        return match["team2"]["name"]
    else:
        return None
        
    
    
   # Given a dictionary containing information about a match,
    #return the name of the winner (or None in the case of a tie)
    
    


# In[19]:


# Run this cell without changes
assert find_winner(matches[0]) == "Russia"
assert find_winner(matches[1]) == "Uruguay"
assert find_winner(matches[2]) == None


# In[20]:


# Replace None with appropriate code

for match in matches:
    # Get the name of the winner
    winner = find_winner(match)
    
      # Only proceed to the next step if there was
    if winner:
        # Add 1 to the associated count of wins
        combined_data[winner]["wins"] += 1
 
      
    

# Visually inspect the output to ensure the wins are
# different for different countries
combined_data


# # Analysis of Wins

# # Statistical Summary of Wins

# In[21]:


# Run this cell without changes
import numpy as np

wins = [val["wins"] for val in combined_data.values()]

print("Mean number of wins:", np.mean(wins))
print("Median number of wins:", np.median(wins))
print("Standard deviation of number of wins:", np.std(wins))


# # Visualizations of Wins

# In[22]:


# Run this cell without changes
import matplotlib.pyplot as plt

# Set up figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 7))
fig.set_tight_layout(True)

# Histogram of Wins and Frequencies
ax1.hist(x=wins, bins=range(8), align="left", color="green")
ax1.set_xticks(range(7))
ax1.set_xlabel("Wins in 2018 World Cup")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Wins")

# Horizontal Bar Graph of Wins by Country
ax2.barh(teams[::-1], wins[::-1], color="green")
ax2.set_xlabel("Wins in 2018 World Cup")
ax2.set_title("Wins by Country");


# # 3. Associating Countries with 2018 Population

# In[23]:


# Run this cell without changes
len(population_data)


# In[24]:


# Run this cell without changes
np.random.seed(42)
population_record_samples = np.random.choice(population_data, size=10)
population_record_samples


# # Filtering Population Data

# In[25]:


# Replace None with appropriate code

population_data_filtered = []

for record in population_data:
    # Add record to population_data_filtered if relevant
    if (record["Country Name"] in teams) and (record["Year"] == "2018"):
        population_data_filtered.append(record)
    
len(population_data_filtered) # 27


# In[26]:


# Run this cell without changes
population_record_samples[2]


# # Normalizing Locations in Population Data

# In[27]:


# Run this cell without changes
def normalize_location(country_name):
    """
    Given a country name, return the name that the
    country uses when playing in the FIFA World Cup
    """
    name_sub_dict = {
        "Russian Federation": "Russia",
        "Egypt, Arab Rep.": "Egypt",
        "Iran, Islamic Rep.": "Iran",
        "Korea, Rep.": "South Korea",
        "United Kingdom": "England"
    }
    # The .get method returns the corresponding value from
    # the dict if present, otherwise returns country_name
    return name_sub_dict.get(country_name, country_name)

# Example where normalized location is different
print(normalize_location("Russian Federation"))
# Example where normalized location is the same
print(normalize_location("Argentina"))


# In[28]:


# Replace None with appropriate code

population_data_filtered = []

for record in population_data:
    # Get normalized country name
    normalized_name = normalize_location(record["Country Name"])
    # Add record to population_data_filtered if relevant
    if (normalized_name in teams) and (record["Year"] == "2018"):
        # Replace the country name in the record
        record["Country Name"] = normalized_name
        # Append to list
        population_data_filtered.append(record)
    
len(population_data_filtered)


# # Type Conversion of Population Data

# In[29]:


# Run this cell without changes
population_data_filtered[0]


# In[30]:


# Run this cell without changes
combined_data


# In[31]:


for record in population_data_filtered:
    # Convert the population value from str to int
    record["Value"] = int(record["Value"])
    
# Look at the last record to make sure the population
# value is an int
population_data_filtered[-1]


# In[32]:


# Replace None with appropriate code
for record in population_data_filtered:
    # Extract the country name from the record
    country = record["Country Name"]
    # Extract the population value from the record
    population = record["Value"]
    # Add this information to combined_data
    combined_data[country]["population"] = population 

# Look combined_data
combined_data


# In[33]:


# Run this cell without changes
assert type(combined_data["Uruguay"]) == dict
assert type(combined_data["Uruguay"]["population"]) == int


# # Analysis of Population

# In[34]:


# Run this cell without changes
populations = [val["population"] for val in combined_data.values()]

print("Mean population:", np.mean(populations))
print("Median population:", np.median(populations))
print("Standard deviation of population:", np.std(populations))


# # Visualizations of Population

# In[35]:


# Run this cell without changes

# Set up figure and axes
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 7))
fig.set_tight_layout(True)

# Histogram of Populations and Frequencies
ax1.hist(x=populations, color="blue")
ax1.set_xlabel("2018 Population")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Population")

# Horizontal Bar Graph of Population by Country
ax2.barh(teams[::-1], populations[::-1], color="blue")
ax2.set_xlabel("2018 Population")
ax2.set_title("Population by Country");


# # 4. Analysis of Population vs. Performance

# In[36]:


# Run this cell without changes
np.corrcoef(wins, populations)[0][1]


# # Data Visualization

# In[37]:


# Run this cell without changes

# Set up figure
fig, ax = plt.subplots(figsize=(8, 5))

# Basic scatter plot
ax.scatter(
    x=populations,
    y=wins,
    color="gray", alpha=0.5, s=100
)
ax.set_xlabel("2018 Population")
ax.set_ylabel("2018 World Cup Wins")
ax.set_title("Population vs. World Cup Wins")

# Add annotations for specific points of interest
highlighted_points = {
    "Belgium": 2, # Numbers are the index of that
    "Brazil": 3,  # country in populations & wins
    "France": 10,
    "Nigeria": 17
}
for country, index in highlighted_points.items():
    # Get x and y position of data point
    x = populations[index]
    y = wins[index]
    # Move each point slightly down and to the left
    # (numbers were chosen by manually tweaking)
    xtext = x - (1.25e6 * len(country))
    ytext = y - 0.5
    # Annotate with relevant arguments
    ax.annotate(
        text=country,
        xy=(x, y),
        xytext=(xtext, ytext)
    )


# In[ ]:




