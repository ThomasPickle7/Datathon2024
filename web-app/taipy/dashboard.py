from taipy.gui import Gui, notify
import pandas as pd 
import numpy as np
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
#masthead of the GUI
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
#about section, contains interactive image selection and info about our project
about = """
<|layout|columns 2 1|
<| 
<|{sample_text}|>
<br></br>
|>

<|
<center>
<|{img_map}|image|height=500px|width=750px|on_action=image_action|>
</center>
<br></br>
<|layout|columns 1 1 1 1 1 1 1 1|

<|
<|button|label=USA|on_action=button_action_map|id=USA|>
|>

<|
<|button|label=HOU|on_action=button_action_map|id=HOU|>
|>

<|
<|button|label=SEA|on_action=button_action_map|id=SEA|>
|>
 
<|
<|button|label=NYC|on_action=button_action_map|id=NYC|>
|>

<|
<|button|label=CHI|on_action=button_action_map|id=CHI|>
|>

<|
<|button|label=ATL|on_action=button_action_map|id=ATL|>
|>

<|
<|button|label=LAX|on_action=button_action_map|id=LAX|>
|>
 
<|
<|button|label=SFO|on_action=button_action_map|id=SFO|>
|>

|>
|>
|>

"""

# the upload section, users can upload and vizualize raw data, as well as select a model to run their data through for analysis
upload = """
<|layout|columns = 1 1|
<|
## Upload your sleep data for analysis!
<br/>
<center>
<|{path}|file_selector|label=Select File|on_action=select_action|extensions=.npy,.npz|drop_message=Select your weather data!|>
</center>
## Select model of analysis
<|layout|columns = 4*1|

<|
<|button|label=XGB|on_action=button_action_model|id=XGB|>
|>

<|
<|button|label=DNN|on_action=button_action_model|id=DNN|>
|>

<|
<|button|label=CNN|on_action=button_action_model|id=CNN|>
|>

<|
<|button|label=FRQ|on_action=button_action_model|id=FRQ|>
|>

|>
<|{txt_model}|>
<br></br>
<|button|label=Run on Model|on_action=button_action_run|id=RUN|>
|>

<|
##Data Visualization
<|{dataset}|chart|mode=lines|x=Date|y[1]=MinTemp|y[2]=MaxTemp|color[1]=blue|color[2]=red|>
<|{img_model}|image|height=500px|width=500px|on_action=image_action|>
|>
|>

"""
# Results section, allows users to view and download the results of their selected model.
results = """
<br></br>
#This is the Results section
<br></br>
<|{dataset}|chart|mode=lines|x=Date|y[1]=MinTemp|y[2]=MaxTemp|color[1]=blue|color[2]=red|>
"""
#ackowledgementsection, contains citations/references and whatnot
acknowledge = """
<br></br>
#This is The Acknowledgements section
<|{sample_text}|>
"""


def image_action(state):
    # links images to a website
    webbrowser.open("https://profiles.rice.edu/faculty/joseph-young")

def get_data(path: str):
    """
    takes in a numpy file, converts to a 2d array
    """
    dataset = pd.read_csv(path)
    dataset["Date"] = pd.to_datetime(dataset["Date"]).dt.date
    return dataset


#slight change "getting rid of 26files"
#made it to pd.read_csv

def load_xgb_files(file_path, **kwargs):
    # Initialize empty lists to store features and labels
    all_X = []
    all_y = []
    import pandas as pd
    counter = 0

    X_train = pd.read_csv(file_path)
    y_train = pd.read_csv(file_path)
    y_train = [int(i) - 1 for i in y_train]
    # # if decimate is true, only use every fifth sample for waking state (y=0)

        #THEO: I am not sure what kwargs do.
    if kwargs.get('decimate'):
        for i in range(len(X_train)):
            if(y_train[i] != 0 or counter == 0):
                all_X.append(X_train[i].transpose())
                for j in range(3000):
                    all_y.append(y_train[i])
            counter += 1
            counter %= 5
    else:
        for i in range(len(y_train)):
            all_X.append(X_train[i].transpose())
            for j in range(3000):
                all_y.append(y_train[i])

        #THEO: is this fine?
        all_X = pd.DataFrame(data=all_X)
        #want to create column names
        all_X.columns = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental','Temp rectal']
        all_y = np.array(all_y)

    return all_X, all_y, y_train










def select_action(state, var_name, value):
    """
    changes current raw dataset to the uploaded file
    """
    state.dataset = get_data(state.path)

#not applicable, remove/modify in final implementation
def start_date_onchange(state, var_name, value):
    state.start_date = value.date()

#not applicable, remove/modify in final implementation
def end_date_onchange(state, var_name, value):  
    state.end_date = value.date()

#not applicable, remove/modify in final implementation
def filter_by_date_range(dataset, start_date, end_date):
    mask = (dataset['Date'] > start_date) & (dataset['Date'] <= end_date)
    return dataset.loc[mask]


def button_action_run(state):
    """
    when the run button is pressed, the currently toggled function is called on the raw_dataset.
    """

    state.model_dataset = state.run

def button_action_map(state, id):
    """
    When a map button is pressed, the above image is updated depending on the specific button pressed
    """
    assets= {
        'USA': "images/maps/USA.jpeg",
        'HOU': "images/maps/HOU.jpeg",
        'SEA': "images/maps/SEA.jpeg",
        'NYC': "images/maps/NYC.jpeg",
        'CHI': "images/maps/CHI.jpeg",
        'ATL': "images/maps/ATL.jpeg",
        'LAX': "images/maps/LAX.jpeg",
        'SFO': "images/maps/SFO.jpeg",
    }
    state.img_map = assets[id] # changes the dataset to reflect dates

def button_action_model(state, id):
    """
    When a model button is pressed, the img_model and txt_model should be updated accordingly, and
    the function used when running the model should be updated accordingly.
    """
    imgs= {
        'XGB': "images/diagrams/XGB.png",
        'DNN': "images/diagrams/DNN.png",
        'CNN': "images/diagrams/CNN.png",
        'FRQ': "images/diagrams/FRQ.png"
    }
    txt = {
        'XGB': "XGB",
        'DNN': "DNN",
        'CNN': "CNN",
        'FRQ': "FRQ"
    }
    func = {
        'XGB': "XGB function",
        'DNN': "DNN function",
        'CNN': "CNN function",
        'FRQ': "FRQ function"
    }
    state.img_model = imgs[id] # changes the image for the model
    state.txt_model = txt[id] # changes the displayed text for the model
    state.run_func = func[id] # changes the run function based on button ID

def download(state):
    """
    when pressed, downloads a csv of the modeled data form teh raw inputs
    """
    state.dataset.to_csv('download.csv')


img_map = "images/maps/USA.jpeg"
img_model = "images/joe.png"
txt_model = "howdy howdy howdy"
logo = "images/joe.png"


dataset = get_data("data/weather.csv")
start_date = datetime.date(2008, 12, 1)
end_date = datetime.date(2017, 6, 25)
download_data = "download.csv"
path = "data/weather.csv"
page = masthead+about+upload+results+acknowledge
# put default function (XGB) here when its integrated
run_func = ""

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