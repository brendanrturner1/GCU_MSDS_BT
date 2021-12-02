#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Welcome! Please note this project is not yet complete.
# Also, this project may take some time run in its completeness. 


# In[2]:


# Import packages
import cfbd
import datetime
import numpy as np
import pandas as pd

# Configure API key authorization: ApiKeyAuth
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'RQoXyuu3evY+/fg5P6ZSZtk4XsdaTkOUWj5DUzin3/FNH5my98yGypkZRe+sR6eo'
configuration.api_key_prefix['Authorization'] = 'Bearer'

# Load APIs
api_config = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_config)
games_api = cfbd.GamesApi(api_config)
drives_api = cfbd.DrivesApi(api_config)
conferences_api = cfbd.ConferencesApi(api_config)


# In[3]:


# Build Drive Dataframe
drives = []

for year in range(2003, 2009):  
    response = drives_api.get_drives(year=year)
    drives = [*drives, *response]


# In[4]:


# Filter dataframe to only collect relevant information

to_list = ['INT','FUMBLE','INT TD','FUMBLE TD']

drivedf = [
    dict(
         # Score info
         offense = d.offense,
         off_score = d.end_offense_score,
         defense = d.defense,
         def_score = d.end_defense_score,
         change = d.end_offense_score - d.start_offense_score,
         drive_result = d.drive_result,
         
         # Yards Per Play
         yards = d.yards,
         plays = d.plays,
         YPP = 0,

         # Start Field Position
         drive_start = d.start_yards_to_goal,

         # Green / Red Zones
         drive_end = d.end_yards_to_goal,
         green = np.where(d.end_yards_to_goal < 41, True, False),
         red = np.where(d.end_yards_to_goal < 21, True, False),
         
         # Turnovers & FGs      
         turnover = np.where(d.drive_result in to_list, True, False),

         # Miscellaneous 
         start_defense_score = d.start_defense_score,
         start_offense_score = d.start_offense_score,
         start_period = d.start_period,
         is_home_offense = d.is_home_offense,      
         
    ) for d in drives]


# In[5]:


# Convert and Clean Dataframe
df = pd.DataFrame.from_records(drivedf).dropna()
df = df.drop(df[df['plays']==0].index) # drop null drives
df['YPP'] = round(df['yards']/df['plays'],3) # compute YPP
df = df.drop(df[df['drive_result'] == 'KICKOFF'].index) # eliminate kickoffs
df = df.drop(df[df['drive_result'] == 'Uncategorized'].index) # eliminate nulls
df['drive_result'] = df['drive_result'].replace('PASSING TD TD','PASSING TD')
df['drive_result'] = df['drive_result'].replace('RUSHING TD TD','RUSHING TD')

# Show examples
df.head(8)


# In[6]:


# Create database of teams
teams_list = []
for i in df['offense'].tolist():
    if i not in teams_list:
        teams_list.append(i)
    else:
        pass
rank = pd.DataFrame(teams_list,columns = ['Team'])
# Set starting value for each team
rank['Score']=1500


# In[7]:


# Model Formula
from statistics import mean

def elo(row):
    o = row.iloc[0]; d = row.iloc[2];
    ypp = row.iloc[8]; to = row.iloc[13];
    ds = row.iloc[9]; r = row.iloc[12];
    g = row.iloc[11]
    ypp=(ypp+20)/100; ds=ds/100;
    if ypp<-.2: 
        ypp=0
    if ypp>.8: 
        ypp=1
        
    x1 = ((ypp**2.1)/(ypp**2.1+.003))
    x2 = ((ds**-6)/(ds**-6+30))
    
    if g == "True":
        x3 = 0.85
    else:
        x3 = 0.15
    if r == "True":
        x4 = 0.95
    else:
        x4 = 0.05
    if to == "True":
        x5 = 0.01
    else:
        x5 = 0.99
    new = mean(2*x1,x2,x3,x4,x5)
    no = rank['Team'].str.match(str(o))
    np = rank['Team'].str.match(str(d))


# In[8]:


def elo(q):
    
    o = q.iloc[0]; d = q.iloc[2];
    ypp = q.iloc[8]; to = q.iloc[13];
    ds = q.iloc[9]; r = q.iloc[12];
    g = q.iloc[11]
    ypp=(ypp+20)/100; ds=ds/100;
    if ypp<-.2: 
        ypp=0
    if ypp>.8: 
        ypp=1
    if ds==0:
        ds=0.1
        
    x1 = (((ypp-.0001)**12) / ((ypp-.0001)**12 + 0.00000008))
    x2 = ((ds**-6)/(ds**-6+30))
    if g == "True":
        x3 = 0.75
    else:
        x3 = 0.25
    if r == "True":
        x4 = 0.85
    else:
        x4 = 0.15
    if to == "True":
        x5 = 0.1
    else:
        x5 = 0.9

    new = mean([x1,x1,x1,x2,x2,x3,x4,x5])
    no = rank.loc[rank['Team']==o]
    nd = rank.loc[rank['Team']==d]
    
    noo = no.iloc[0,1]
    ndd = nd.iloc[0,1]

    exp = 1/(1+(10**((noo-ndd)/-400)))
    change = 50*(new-exp)

    o_i = rank.index.get_loc(rank.index[rank['Team'] == o][0])
    d_i = rank.index.get_loc(rank.index[rank['Team'] == d][0])
    o_r = rank.iloc[o_i][1]
    d_r = rank.iloc[d_i][1]

    tt1 = round(o_r + change,2)
    tt2 = round(d_r - change,2)

    rank.loc[rank['Team'] == o,"Score"] = tt1
    rank.loc[rank['Team'] == d,"Score"] = tt2


# In[9]:


for index, row in df.iterrows():
    elo(row)


# In[10]:


rank = rank.sort_values(by=['Score'],ascending=False)
rank.head(25)


# In[11]:


# Running the following scripts produces a working GUI, 
# but some features are still being implemented. 


# In[12]:


import PySimpleGUI as sg
import csv

def full_rank():
    filename = r'C:\Users\giant\OneDrive\Documents\cfb_rank.csv'  
    data = list(rank)
    header_list = ['Team', 'Score']
    sg.set_options(element_padding=(2, 2))
    layout = [[sg.Table(values=data,
                        headings=header_list,
                        max_col_width=50,
                        auto_size_columns=True,
                        justification='right',
                        #autoscroll = True,
                        #alternating_row_color='lightblue',
                        num_rows=(len(data)))]]
    window1 = sg.Window('Full Rankings', layout, grab_anywhere=True)
    event, values = window1.read()
    window1.close()

def this_pred():
    filename = r'C:\Users\giant\OneDrive\Documents\cfb_this.csv'  
    data = []
    header_list = []
    with open(filename, "r") as infile:
        reader = csv.reader(infile)
        header_list = next(reader)
        data = list(reader)  
    sg.set_options(element_padding=(2, 2))
    layout = [[sg.Table(values=data,
                        headings=header_list,
                        max_col_width=50,
                        auto_size_columns=True,
                        justification='right',
                        #alternating_row_color='lightblue',
                        num_rows=min(len(data), 20))]]
    window2 = sg.Window("This Week's Predictions", layout, grab_anywhere=True)
    event, values = window2.read()
    window2.close()
    
def last_pred():
    filename = r'C:\Users\giant\OneDrive\Documents\cfb_last.csv'  
    data = []
    header_list = []
    with open(filename, "r") as infile:
        reader = csv.reader(infile)
        header_list = next(reader)
        data = list(reader)  
    sg.set_options(element_padding=(2, 2))
    layout = [[sg.Table(values=data,
                        headings=header_list,
                        max_col_width=50,
                        auto_size_columns=False,
                        justification='right',
                        #alternating_row_color='lightblue',
                        num_rows=min(len(data), 20))]]
    window3 = sg.Window("Last Week's Predictions", layout, grab_anywhere=True)
    event, values = window3.read()
    window3.close()
    
def about():
    layout = [[sg.B('User Guide'), sg.Text(size=(15,1))],
              [sg.B('Questions?'), sg.Text(size=(15,1))],
              [sg.B('Back'), sg.Text(size=(15,1))]]
    
    window4 = sg.Window("About", layout, grab_anywhere=True)
    while True:  # Event Loop
        event, values = window4.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "Back":
            break
        if event == "User Guide":
            guide()
        if event == "Questions?":
            questions()
    
    window4.close()
    
def guide():
    # NOTE: text here is subject to change
    layout = [[sg.Text('This formula uses a ranking system that differs from the many existing '), sg.Text(size=(15,1))],
              [sg.Text('basic systems in that it does not rely on wins and losses to determine '), sg.Text(size=(15,1))],
              [sg.Text('its places. Instead, our model will evaluate how a team played, since '), sg.Text(size=(15,1))],
              [sg.Text('it stands to reason that teams which consistently play well are likely '), sg.Text(size=(15,1))],
              [sg.Text('to repeat their high level of play, contrasted with teams who might have '), sg.Text(size=(15,1))],
              [sg.Text('their performance vary between games but experience bouts of luck which '), sg.Text(size=(15,1))],
              [sg.Text('ensure they still win their games. Another difference from many public '), sg.Text(size=(15,1))],
              [sg.Text('attempts to solve this problem is that our model is designed to be '), sg.Text(size=(15,1))],
              [sg.Text('predictive and forward-facing, and not be a résumé evaluation tool '), sg.Text(size=(15,1))],
              [sg.Text('that rewards teams for playing tougher opponents. '), sg.Text(size=(15,1))]]

    window5 = sg.Window("User Guide", layout, grab_anywhere=True)
    event, values = window5.read()
    window5.close()
    
def sent():
        # NOTE: text here is subject to change
    layout = [[sg.Text('Your message has been sent!'), sg.Text(size=(15,1))]]
              
    window6 = sg.Window("Thank You!", layout, grab_anywhere=True)
    event, values = window6.read()
    window6.close()
    
def questions():
    layout =[[sg.Text('Message will automatically be sent to the developer. We value your input!')],
            [sg.Text('Return Address:'), sg.InputText()],
            [sg.Text('Subject'), sg.InputText()],
            [sg.Multiline(size=(30, 5), key='textbox')],
            [sg.Button('Send'), sg.Button('Back')]]  # identify the multiline via key option

        # NOTE: Full email functionality has not yet been implemented 
    window7 = sg.Window("Questions", layout, grab_anywhere=True)
    while True:  # Event Loop
        event, values = window7.read()
        if event == sg.WIN_CLOSED or event == "Back":
            break
        if event == "Send":
            sent()
    
    window7.close()
    
def imp():
    # NOTE: Full URL and file data importing is currently buggy  
    # so this feature is temporarily replaced by a dummy window 
    # representing what the final product will look like
    layout =[[sg.Text('To import data from URL, please enter link here:')],
            [sg.InputText()],
            [sg.Button('Get URL')],
            [sg.Text('To import data from file, please click the button below:')],
            [sg.Button('Choose file')], 
            [sg.Button('Back')]]  # identify the multiline via key option

        # NOTE: Full email functionality has not yet been implemented 
    window8 = sg.Window("Import Data", layout, grab_anywhere=True)
    while True:  # Event Loop
        event, values = window8.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "Back":
            break
    
    window8.close()


# In[13]:


def main():
    sg.theme('DarkGreen5')

    layout = [[sg.Text('Welcome! Please choose an option to continue:'), sg.Text(size=(15,1))],
              [sg.B("Full Team Rankings")],
              [sg.B("This Week's Predictions")],
              [sg.B("Last Week's Predictions")],
              [sg.B("About")],
              [sg.B("Import Data")],
              [sg.B("Exit")]]

    win = sg.Window('BT College Football Power Ratings', layout)

    while True:  # Event Loop
        event, values = win.read()
        #print(event, values)
        if event == sg.WIN_CLOSED or event == "Exit":
            break
        if event == "Full Team Rankings":
            full_rank()
        if event == "This Week's Predictions":
            this_pred()
        if event == "Last Week's Predictions":
            last_pred()
        if event == "About":
            about()
        if event == "Import Data":
            imp()

    win.close()
main()

