from taipy.gui import Gui, notify
import pandas as pd 
import webbrowser
import datetime

"""
<center>
<h2>Dataset</h2><|{download_data}|file_download|on_action=download|>
</center>
<center>
<|{dataset}|table|page_size=10|height=500px|width=65%|>
</center>
"""

"""
**Starting Date**\n\n<|{start_date}|date|not with_time|on_change=start_date_onchange|>
<br/><br/>
**End Date**\n\n<|{end_date}|date|not with_time|on_change=end_date_onchange|>
<br/><br/>*/
"""

masthead = """
<|layout|columns = 7 1|
<|
#Retake your Sleep with *PowerNap*!
|>
<|
<center>
<|{logo}|image|height=100px|width=90px|on_action=image_action|>
</center>
|>
|>
"""

about = """
<|layout|column 2 1|
<| 
<|{sample_text}|>
|>

<|
<center>
<|{logo}|image|height=500px|width=500px|on_action=image_action|>
</center>
<|layout|column 8*1|

<|
<|button|label=USA|on_action=button_action_image|id=USA|>
|>

<|
<|button|label=HOU|on_action=button_action_image|id=HOU|>
|>

<|
<|button|label=SEA|on_action=button_action_image|id=SEA>
|>
 
<|
<|button|label=NYC|on_action=button_action_image|id=NYC|>
|>

<|
<|button|label=CHI|on_action=button_action_image|id=CHI|>
|>

<|
<|button|label=ATL|on_action=button_action_image|id=ATL|>
|>

<|
<|button|label=LAX|on_action=button_action_image|id=LAX|>
|>
 
<|
<|button|label=SFO|on_action=button_action_image|id=SFO|>
|>

|>
|>

"""

upload = """
<|layout|column = 1 1|
<|
## Upload your sleep data for analysis!
<br/>
<center>
<|{path}|file_selector|label=Select File|on_action=select_action|extensions=.csv,.xlsx|drop_message=Select your weather data!|>
</center>
## Select model of analysis
<|layout|column = 4*1|

<|
<|button|label=XGB|on_action=button_action_image|id=KGB>
|>

<|
<|button|label=DNN|on_action=button_action_image|id=DNN>
|>

<|
<|button|label=CNN|on_action=button_action_image|id=CNN|>
|>

<|
<|button|label=FRQ|on_action=button_action_image|id=FRQ|>
|>

|>
|>
<|
##Data Visualization
<|{dataset}|chart|mode=lines|x=Date|y[1]=MinTemp|y[2]=MaxTemp|color[1]=blue|color[2]=red|>
|>
|>

"""

results = """
<br></br>
#This is the Results section
<br></br>
<|{dataset}|chart|mode=lines|x=Date|y[1]=MinTemp|y[2]=MaxTemp|color[1]=blue|color[2]=red|>
"""

acknowledge = """
<br></br>
#This is The Acknowledgements section
<|{sample_text}|>
"""


def image_action(state):
    webbrowser.open("https://profiles.rice.edu/faculty/joseph-young")

def get_data(path: str):
    dataset = pd.read_csv(path)
    dataset["Date"] = pd.to_datetime(dataset["Date"]).dt.date
    return dataset

def select_action(state, var_name, value):
    state.dataset = get_data(state.path)

def start_date_onchange(state, var_name, value):
    state.start_date = value.date()

def end_date_onchange(state, var_name, value):  
    state.end_date = value.date()

def filter_by_date_range(dataset, start_date, end_date):
    mask = (dataset['Date'] > start_date) & (dataset['Date'] <= end_date)
    return dataset.loc[mask]

def button_action(state):
    state.dataset = filter_by_date_range(dataset, state.start_date, state.end_date) # changes the dataset to reflect dates
    notify(state, "info", "updated date range from {} to {}".format(start_date.strftime("%m/%d/%Y"), end_date.strftime("%m/%d/%Y")))

def button_action_image(state, id):
    assets= {
        'USA': "images/maps/USA.png",
        'HOU': "images/maps/HOU.png",
        'SEA': "images/maps/SEA.png",
        'NYC': "images/maps/NYC.png",
        'CHI': "images/maps/CHI.png",
        'ATL': "images/maps/ATL.png",
        'LAX': "images/maps/LAX.png",
        'SFO': "images/maps/SFO.png",
        'XGB': "images/diagrams/XGB.png",
        'DNN': "images/diagrams/DNN.png",
        'CNN': "images/diagrams/CNN.png",
        'FRQ': "images/diagrams/FRQ.png"
    }
    state.img_map = assets[id] # changes the dataset to reflect dates

def download(state):
    state.dataset.to_csv('download.csv')

img_map = "images/maps/USA.png"
logo = "images/joe.png"
dataset = get_data("data/weather.csv")
start_date = datetime.date(2008, 12, 1)
end_date = datetime.date(2017, 6, 25)
download_data = "download.csv"
path = "data/weather.csv"
page = masthead+about+upload+results+acknowledge

sample_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
 sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
 Tincidunt vitae semper quis lectus nulla. Sagittis orci a scelerisque 
 purus semper eget duis at. Suspendisse potenti nullam ac tortor. Dolor 
 sit amet consectetur adipiscing. Id velit ut tortor pretium viverra suspendisse potenti nullam.
   Lectus mauris ultrices eros in cursus turpis massa tincidunt dui. Et molestie ac feugiat sed. Justo donec enim diam vulputate 
   ut pharetra sit amet. Et netus et malesuada fames ac turpis egestas. Massa sed elementum tempus egestas sed sed risus. Morbi tempus 
   iaculis urna id volutpat lacus laoreet non. Volutpat est velit egestas dui id ornare arcu odio. Dis parturient montes nascetur ridiculus
     mus mauris vitae ultricies leo. Ut sem viverra aliquet eget sit amet tellus cras. Id aliquet risus feugiat in ante. Sociis natoque penatibus
       et magnis dis parturient montes. Vestibulum mattis ullamcorper velit sed ullamcorper morbi.

Fermentum iaculis eu non diam phasellus vestibulum lorem sed risus. Sed euismod nisi porta lorem mollis aliquam ut porttitor leo. Eget 
mi proin sed libero enim. Bibendum enim facilisis gravida neque convallis a. Et netus et malesuada fames ac. Malesuada pellentesque elit
 eget gravida cum sociis. Etiam sit amet nisl purus in mollis nunc sed id. Convallis a cras semper auctor neque vitae. Tellus mauris a diam
   maecenas sed enim ut sem. Eu augue ut lectus arcu."
"""
Gui(page = page).run()