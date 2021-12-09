#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Welcome! Please note this project may take some time run in its completeness.


# In[2]:


pip install cfbd


# In[3]:


pip install natsort


# In[4]:


pip install PySimpleGUI


# In[5]:


# Import packages
import cfbd
import datetime
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
import PySimpleGUI as sg
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


# In[6]:


# Get 2016 data

# Get drive data
drives = []
for year in range(2016,2017):  
    response = drives_api.get_drives(year=year)
    drives = [*drives, *response]

# Filter dataframe to only collect relevant information
to_list = ['INT','FUMBLE','INT TD','FUMBLE TD']
drivedf = [
    dict(
         # Score info
         offense = d.offense,
         off_score = d.end_offense_score,
         defense = d.defense,
         def_score = d.end_defense_score,
         change = d.end_offense_score - d.start_offense_score - (d.end_defense_score - d.start_defense_score),
         lead = d.end_offense_score - d.end_defense_score,
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
         #start_def_score = d.start_defense_score,
         #start_off_score = d.start_offense_score,
         #start_qtr = d.start_period,
         is_home_off = d.is_home_offense, 
         margin = d.end_offense_score - d.end_defense_score,
         game_id = d.game_id,
         offconf = d.offense_conference,
         defconf = d.defense_conference,
         dn = d.drive_number

    ) for d in drives]

# Convert and Clean Dataframe
df = pd.DataFrame.from_records(drivedf).dropna()
df = df.drop(df[df['plays']==0].index) # drop null drives
df['YPP'] = round(df['yards']/df['plays'],3) # compute YPP
df = df.drop(df[df['drive_result'] == 'KICKOFF'].index) # eliminate kickoffs
df = df.drop(df[df['drive_result'] == 'Uncategorized'].index) # eliminate nulls
df['drive_result'] = df['drive_result'].replace('PASSING TD TD','PASSING TD')
df['drive_result'] = df['drive_result'].replace('RUSHING TD TD','RUSHING TD')


# In[7]:


#df.to_csv(r'C:\Users\giant\OneDrive\Documents\drives.csv', encoding='utf-8',index=False)


# In[8]:


# Create database of conferences
FBS_conf = ['Sun Belt','Conference USA','Western Athletic','Pac-12','Big 12','Mid-American',
            'Big Ten','American Athletic','Mountain West','SEC','FBS Independents','ACC']
Power_conf = ['Pac-12','Big 12','Big Ten','SEC','ACC']


# In[9]:


# Create database of teams
teams_list = []
for i in df['offense'].tolist():
    if i not in teams_list:
        teams_list.append(i)
    else:
        pass


# In[10]:


# Create database of only FBS teams
FBS_teams = []
for index,row in df[['offense','offconf']].iterrows():
    if row['offconf'] in FBS_conf and row ['offense'] not in FBS_teams:
        FBS_teams.append(row['offense'])
    else:
        pass
Power_teams = []
for index,row in df[['offense','offconf']].iterrows():
    if row['offconf'] in Power_conf and row ['offense'] not in Power_teams:
        Power_teams.append(row['offense'])
    else:
        pass


# In[11]:


rank = pd.DataFrame(teams_list,columns = ['Team'])
# Set starting value for each team
rank['Score']=1475
pd.set_option("display.max_rows", None)
for i in rank.itertuples():
    if i[1] in Power_teams:
        rank.loc[i[0],['Score']] = 1525
#rank.to_csv(r'C:\Users\giant\OneDrive\Documents\cfb_rank.csv', encoding='utf-8',index=False)


# In[12]:


# Model Formula
from statistics import mean

# Define Game Control Metric
def control(d):
    C = []
    for i in d.itertuples():
        lead = int(i[6])
        con = 1/(1+(10**((lead**3/-5000))))
        C.append(round(con,2))
    d['Control']=C  

# Define model update function
def elo(g):
    game =  df.loc[df['game_id']==g]
    o = game.iloc[0,0] #team that starts with the ball
    d = game.iloc[0,2] #team that starts on defense
    oo = game.loc[game['offense']==o]; dd = game.loc[game['offense']==d]
    control(oo)
    control(dd)
    OGC = round(mean(oo['Control'].tolist()[3:])+0.1,3) # Game Control
    
    # Get team info
    oe = game.iloc[-1,1]; de = game.iloc[-1,3]
    ot = rank.loc[rank['Team']==o]; dt = rank.loc[rank['Team']==d]
    ott = ot.iloc[0,0]; dtt = dt.iloc[0,0]
    otr = ot.iloc[0,1]; dtr = dt.iloc[0,1] 
    home = game.iloc[-1,15]
    dif = 0
    o_won = True
    
    if o==game.iloc[-1,0]:#team that started w/ball also ended w/ball
        if oe>de:#team that started won
            o_won = True
        else:#team that started lost
            o_won = False
    else:#team that started did NOT end
        if de>oe:#team that started won
            o_won = True
        else:#team that started lost
            o_won = False
    
    c=0
    # Check if we predicted the result correctly
    if o_won == True:
        if otr>dtr:
            c = 1
        else:
            c = 0
    if o_won == False:
        if dtr>otr:
            c = 1
        else:
            c = 0
    
    if o_won == True: #team A won
        w=1
        k=(19+(3*(OGC**.9)-1))*(np.log(abs(oe-de))+1)*(2.2/(2.2+(otr-dtr)/1000))
        if home == True:
            dif = 65
        else:
            dif = -65
        we = 1/(1+(10**(-((otr-dtr)+dif)/400)))
        newo = round(otr + k*(w-we),1); newd = round(dtr - k*(w-we),1)
    else: #team B won
        w=1
        k=(19+(3*((1-OGC)**.9)-1))*(np.log(abs(de-oe))+1)*(2.2/(2.2+(dtr-otr)/1000))
        if home == True:
            dif = -65
        else:
            dif = 65
        we = 1/(1+(10**(-((dtr-otr)+dif)/400)))
        newo = round(otr - k*(w-we),1); newd = round(dtr + k*(w-we),1)
        
    # Update rankings
    rank.loc[rank['Team']==o,'Score'] = newo
    rank.loc[rank['Team']==d,'Score'] = newd
    
    # Debugging
    if abs(de-oe)<0.1 or abs(oe-de)<0.1:
        print(g)
    else:
        pass   
    
    return c


# In[13]:


# Build games list
games = []
for i in df['game_id'].tolist():
    if i not in games:
        games.append(i)
pd.options.mode.chained_assignment = None  # default='warn'

# Fix known 
df.at[20173,'off_score'] = 37
w=0
for i in games:
    c=elo(i)
    if c==1:
        w=c+w


# In[14]:


from natsort import index_natsorted
rank = rank.sort_values(by='Score',key=lambda x: np.argsort(index_natsorted(rank['Score'],reverse=True)))
rank = rank.reset_index(drop=True)
rank.index += 1
rank
#rank.to_csv(r'C:\Users\giant\OneDrive\Documents\cfb_rank.csv', encoding='utf-8',index=False)


# In[15]:


wins=100*round(w/len(games),2)
rank2 = rank.to_csv("rank.csv")
rank = pd.read_csv("rank.csv")
rank


# In[16]:


# Define Box Score Formula

def box(a,b):
    ta = rank.loc[rank['Team']==a]
    tb = rank.loc[rank['Team']==b]
    ra = ta.iloc[0,2]; rb = tb.iloc[0,2]
    prob_a = 100*round(1/(1+10**((rb-ra)/400)),2)
    prob_b = 100-prob_a
    if prob_a>prob_b:
        spread_a = 60*(prob_a/100)**3-6*(prob_a/100)**2-6
        spread_a = "-" + str(round(spread_a*2)/2)
        spread_b = "+" + str(spread_a[1:])
    else: 
        spread_b = 60*(prob_b/100)**3-6*(prob_b/100)**2-6
        spread_b = "-" + str(round(spread_b*2)/2)
        spread_a = "+" + str(spread_b[1:])
    data = [[a,prob_a,spread_a],[b,prob_b,spread_b]]
    return data


# In[17]:


L = []
for j in games[-10:]:
    game =  df.loc[df['game_id']==j]
    o = game.iloc[0,0] #team that starts with the ball
    d = game.iloc[0,2] #team that starts on defense
    x = box(o,d)
    for i in x:
        L.append(i)
    L.append([" ", " ", " "])
test = pd.DataFrame(L,columns=['Team','Win %','Spread'])
test.drop(test.tail(1).index,inplace=True)
#test.to_csv(r'C:\Users\giant\OneDrive\Documents\this_pred.csv', encoding='utf-8',index=False)


# In[18]:


def analysis(df,pred,w):
    layout = [[sg.Text('Choose a Data Analysis Method:'), sg.Text(size=(15,1))],
              [sg.B('View Full Rankings'), sg.Text(size=(15,1))],
              [sg.B('View Future Predictions'), sg.Text(size=(15,1))],
              [sg.B('Back'), sg.Text(size=(15,1))]]
    window3 = sg.Window("Data Analysis", layout, grab_anywhere=True)
    while True:  # Event Loop
        event, values = window3.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "Back":
            break
        if event == "View Full Rankings":
            full_rank(df)
        if event == "View Future Predictions":
            future(pred,w)
    window3.close()


# In[19]:


def full_rank(df):
    data = df.values.tolist()
    header_list = ['Rank','Team','Score']
    layout = [[sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(25,25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]]
    
    window1 = sg.Window('Full Rankings', layout, grab_anywhere=True)
    event, values = window1.read()
    window1.close()


# In[20]:


def future(pred,w):
    data = pred.values.tolist()
    header_list = list(pred.columns)
    layout = [[sg.Text("This Season's Prediction Accuracy: "+str(w)+'%'), sg.Text(size=(12,1),font = 'Helvetica 20')],
                [sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(25,25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]]
    
    window2 = sg.Window("This Week's Predictions", layout, grab_anywhere=True)
    event, values = window2.read()
    window2.close()


# In[21]:


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


# In[22]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw(a,b,e,f):
    fig1 = plt.figure(dpi=125)
    x = [e,f]
    y = [int(a),int(b)]
    p1 = plt.bar(x,y,color=['purple','grey'])
    plt.ylabel('Win Probability')
    plt.xlabel('Teams')
    plt.title('Expected Result of '+e+" and "+f+":")
    figure_x, figure_y, figure_w, figure_h = fig1.bbox.bounds
    return (x,y,fig1,figure_x, figure_y, figure_w, figure_h)

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')

def get_team_info(lol):
    pra = lol[0][1]
    prb = lol[1][1]
    spa = lol[0][2]
    spb = lol[1][2]
    return pra,prb,spa,spb

def plots(df):
        
    layout = [[sg.Text('Welcome! Please choose two teams to compare:'), sg.Text(size=(15,1))],
              [sg.Text('Team 1:'), sg.Text(size=(15,1))],
              [sg.Combo(list(df.iloc[:,1]), size=(20,1), default_value='Select', enable_events=True, key='l1')],
              [sg.Text('Team 2:'), sg.Text(size=(15,1))],
              [sg.Combo(list(df.iloc[:,1]), size=(20,1), default_value='Select', enable_events=True, key='l2')],
              [sg.B(' Go! '), sg.Text(size=(15,1))],
              [sg.Text(' ', key='plot', font = 'Helvetica 20')],
              [sg.Canvas(size=(300,300), key='-CANVAS-', pad=(10,10))],
              [sg.B('Back'), sg.Text(size=(15,1))]]
    
    
    window4 = sg.Window("Plots", layout, grab_anywhere=True)
    fig_agg = None
    while True:  # Event Loop
        event, values = window4.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "Back":
            break
        if event == " Go! ":
            if values['l1']=='Select' or values['l2']=='Select':
                sg.popup('Error!','Please Select Two Teams!')
            else:
                q = box(values['l1'],values['l2'])
                a,b,c,d = get_team_info(q)
                if fig_agg is not None:
                    delete_fig_agg(fig_agg)
                _,_,fig1,figure_x, figure_y, figure_w, figure_h = draw(a,b,values['l1'],values['l2'])
                w = str(a); x = str(b); y = str(values['l1']); z = str(values['l2'])
                canvas_elem = window4['-CANVAS-'].TKCanvas
                canvas_elem.Size=(int(figure_w),int(figure_h))
                fig_agg = draw_figure(canvas_elem, fig1)
                if a>b:
                    window4['plot'].update(y+" would be a "+w+"% favorite against "+z)
                elif b>a:
                    window4['plot'].update(y+" would be a "+w+"% underdog against "+z)
                elif a==b:
                    window4['plot'].update(y+" would be evenly matched against "+z)
    
    window4.close()


# In[23]:


def user():
    layout = [[sg.B("Data Analysis")],
              [sg.Text('This gives you the option of viewing the overall rankings,')],
              [sg.Text('or viewing the predictions for games this upcoming week.')],
              [sg.B("  Plots  ")],
              [sg.Text('This allows you to choose any two teams and see how likely the')],
              [sg.Text('model thinks each team would be to win that matchup.')],
              [sg.B("Generate Reports")],
              [sg.Text('This creates a series of figures demonstrating')],
              [sg.Text("the model's performance.")]]

    window6 = sg.Window("User Guide", layout, grab_anywhere=True)
    while True:  # Event Loop
        event, values = window6.read()
        if event == sg.WIN_CLOSED or event == "Back":
            break


# In[24]:


def drawypp(df):
    fig1 = plt.figure(dpi=125)
    x = df.iloc[:,9]
    p1 = plt.hist(x,bins=30,range=[-20,40],color=['blue'])
    plt.ylabel('Frequency')
    plt.xlabel('Yards Per Play')
    plt.title('Histogram of Yards Per Play, Entire Season')
    figure_x, figure_y, figure_w, figure_h = fig1.bbox.bounds
    return (x,fig1,figure_x, figure_y, figure_w, figure_h)

def drawds(df):
    fig2 = plt.figure(dpi=125)
    x2 = df.iloc[:,10]
    p1 = plt.hist(x2,bins=30,color=['lightblue'])
    plt.ylabel('Frequency')
    plt.xlabel('Drive Start')
    plt.title('Histogram of Starting Field Position, Entire Season')
    figure_x2, figure_y2, figure_w2, figure_h2 = fig2.bbox.bounds
    return (x2,fig2,figure_x2, figure_y2, figure_w2, figure_h2)

def reports(df,w):
    layout = [[sg.Text('Generate some Key Performance Metrics:', key='label1', font = 'Helvetica 20')],
              [sg.Text("This Season's Prediction Accuracy: "+str(w)+'%', key='label4', font = 'Helvetica 20')],
              [sg.B('Get YPP'), sg.B('Get SFP'), sg.Text(size=(15,1))],
              [sg.Canvas(size=(80,80), key='-CANVAS1-', pad=(10,10))],
#               [sg.Text('Starting Field Position:', key='label3', font = 'Helvetica 20')],
#               [sg.Canvas(size=(100,100), key='-CANVAS2-', pad=(10,10))],
              [sg.B('Back'), sg.Text(size=(15,1))]]
    
    
    window9 = sg.Window("Plots", layout, grab_anywhere=True)
    fig_agg = None
    while True:  # Event Loop
        event, values = window9.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Back':
            break
        if event == "Get YPP":
            if fig_agg is not None:
                delete_fig_agg(fig_agg)
            _,fig1,figure_x, figure_y, figure_w, figure_h = drawypp(df)
            canvas_elem1 = window9['-CANVAS1-'].TKCanvas
            canvas_elem1.Size=(60,60)
            fig_agg = draw_figure(canvas_elem1, fig1)
        if event == "Get SFP":
            if fig_agg is not None:
                delete_fig_agg(fig_agg)
            _,fig2,figure_x2, figure_y2, figure_w2, figure_h2 = drawds(df)
            canvas_elem2 = window9['-CANVAS1-'].TKCanvas
            canvas_elem2.Size=(60,60)
            fig_agg = draw_figure(canvas_elem2, fig2)
    
    window9.close()


# In[25]:


def about(drives,w):
    layout =[[sg.Button(' User Guide ')],
            [sg.Button('Generate Reports')]]  # identify the multiline via key option

        # NOTE: Full email functionality has not yet been implemented 
    window7 = sg.Window("Help", layout, grab_anywhere=True)
    while True:  # Event Loop
        event, values = window7.read()
        if event == sg.WIN_CLOSED or event == " Back ":
            break
        if event == " User Guide ":
            user()
        if event == "Generate Reports":
            reports(drives,w)
    
    window7.close()


# In[26]:


def main(df,w,pred,drives):
    sg.theme('DarkGreen5')

    layout = [[sg.Text('Welcome! Please choose an option to continue:'), sg.Text(size=(15,1))],
              [sg.B("Data Analysis")],
              [sg.B("  Plots  ")],
              [sg.B("  Help  ")],
              [sg.B("  Exit  ")]]

    win = sg.Window('BT College Football Power Ratings', layout)

    while True:  # Event Loop
        event, values = win.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "  Exit  ":
            break
        if event == "Data Analysis":
            analysis(df,pred,w)
        if event == "  Plots  ":
            plots(df)
        if event == "  Help  ":
            about(drives,w)

    win.close()


# In[27]:


main(rank,wins,test,df)


# In[ ]:





# In[ ]:




