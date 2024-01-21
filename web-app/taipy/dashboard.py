import os
import ML
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
#*PowerNap* - Sleep is Powerful!
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
<|layout|columns = 1 1|
<| 
The connection between sleep data and socio-economic outcomes is multifaceted, and is especially noticeable when viewing the impact of
 sleep on work and school performance. Sleep patterns and duration can significantly influence cognitive functions, productivity, and
  overall well-being, contributing to disparities in performance that are relevant when creating a more equitable society.

Research consistently demonstrates that individuals with sufficient and quality sleep tend to exhibit higher levels of cognitive functioning, 
improved concentration, and enhanced memory retention. Ensuring equitable access to conditions conducive to restful sleep is essential.
 Factors such as work schedules, cultural expectations, and double standards imposed on minority groups can influence sleep patterns
  differently and greatly affect outcomes down the line.

Addressing sleep-related disparities can be a key component of fostering equity in educational and workplace settings. Individuals
 with consistent, quality sleep are more likely to experience academic and professional success. Therefore, promoting 
 sleep awareness and implementing policies that support healthy sleep habits can contribute to leveling the playing field and 
 promoting inclusivity.

Considerations related to DEI should also extend to accommodating diverse sleep needs. Acknowledging and respecting cultural differences, accommodating individuals with varying chronotypes (biological predispositions for different sleep-wake patterns), and providing flexibility in work or academic schedules can create environments that prioritize inclusivity and support overall well-being.

In conclusion, the relationship between sleep data and DEI underscores the importance of recognizing and addressing sleep-related disparities to promote equal opportunities and enhance overall performance in educational and professional settings. Creating inclusive policies and environments that prioritize sleep can contribute to a more equitable and diverse landscape.

We started by searching for a correlation between sleep time and socio-economic outcomes.
When we found a generally positive correlation between hours of sleep and socio-economic 
 status, we then sought out geographic data which tracked the percentage of the population 
 that reported getting less than seven hours of sleep on average. What we found was that lower
  income areas, such as the Second Ward or Kashmere Gardens in Houston, reported a far higher 
  percentage of a lack of sleep.

With PowerNap, we allow users to upload their own sleeping data, where they can view raw EM wave recordings, as well as run that data on our models such that they can interactively analyze their stages of sleep.

<br></br>
|>

<|
<center>
<|{img_map}|image|height=500px|width=750px|on_action=image_action|>
</center>
<br></br>
<|layout|columns = 8*1|gap=2px|

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
preproc = """
<|layout|columns = 1 1|
<| 
##Processing and Analysis
To pre-process data, we conduct a Fast Fourier Transform to pass a frequency domain version of the 3D NumPy array. 
The raw 3D array is taken and cut down from 6 channels to only channels 0 and 1 (index numbering), which are the 
EEG channels; we found EEG measurements to contain artifacts in the frequency domain (alpha, beta, delta, theta, 
and sigma waves) most strongly correlated with different sleep stages; the way we measured each wave’s “contribution”
 per epoch was by taking the power from the frequency range that matched with each different type of wave, then 
 calculating a vector (per epoch) of the power component of each wave type. Then, these vectors were normalized and
concatenated across all epochs into a matrix, size 5x(epochs). We used a 75% training, 25% testing split and passed
testing data to train our XGBoost model with a target type of classification alongside known y-values as labels.
Our model parameters including alpha, beta, gamma, and tree depth were optimized using a grid-search. To classify
 an epoch from the test data, we conduct an FFT, find power in certain spectral bands, and feed this data to our 
 XGBoost model to produce a predicted label. Once each epoch has been processed across a test sample, we scan to 
 post-process out anomalies that we noticed could occur in edge cases that aren’t well modeled by our training 
 data. On our test sample, we scored 85% accurate classification with similar results for data that was in our 
training set and testing set. As such, we are confident that our model is not over-fit and are generally 
pleased with the results.
<br></br>
|>

<|
<center>
<|{img_preproc}|image|height=500px|width=750px|on_action=image_action|>
</center>
<br></br>
<|layout|columns = 4*1|

<|
<|button|label=Node Map|on_action=button_action_preproc|id=eeg|>
|>

<|
<center>
<|button|label=Block Diagram|on_action=button_action_preproc|id=diagram|>
</center>
|>

<|
<center>
<|button|label=Power Spectrum Analysis|on_action=button_action_preproc|id=freq|>
</center>
|>

<|
<center>
<|button|label=Time Domain|on_action=button_action_preproc|id=time|>
</center>
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
<|{path}|file_selector|label=Select File|on_action=select_action|extensions=.npy,.npz|drop_message=Select your EEG data!|>
</center>
## Select model of analysis
<|layout|columns = 3*1|

<|
<|button|label=XGB|on_action=button_action_model|id=XGB|>
|>

<|
<|button|label=DNN|on_action=button_action_model|id=DNN|>
|>

<|
<|button|label=CNN|on_action=button_action_model|id=CNN|>
|>


|>
### Select the desired model for your data, then hit Run to see your hypnogram!
<br></br>
<|button|label=Run|on_action=button_action_run|id=RUN|>
|>

<|
##Data Visualization
<|{raw_dataset_graph}|chart|mode=lines|x=Epoch|y[1]=EEG-Fpz-Cz|y[2]=EEG-Pz-Oz|y[3]=EOG-horizontal|y[4]=Resp-oro-nasal|y[5]=EMG-submental|y[6]=Temp-rectal|color[1]=red|color[2]=orange|color[3]=yellow|color[4]=green|color[5]=blue|color[6]=purple|>
|>
|>

"""
# Results section, allows users to view and download the results of their selected model.
results = """
<br></br>
#Live Results
<br></br>
<|{model_data}|chart|mode=lines|x=Epochs (30s)|y=Sleep Stage|>
"""
#ackowledgementsection, contains citations/references and whatnot
acknowledge = """
<br></br>
#Acknowledgements

Created for Rice Datathon by Team PowerNap: Matthew Karazincir, Theodore Kim, Leo Marek, and Thomas Pickell.

<|{Bureau of Labor Statistics. "Appendix D - USPS State Abbreviations and FIPS Codes." U.S. Bureau of Labor Statistics. September 27, 2005. https://doi.org/https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm.

Bureau of Transportation Statistics. "2016 Noise Data." United States Department of Transportation. August 17, 2023. https://doi.org/https://data-usdot.opendata.arcgis.com/documents/usdot::2016-noise-data/about.

CDC. "500 Cities: Sleeping Less than 7 Hours among Adults Aged >=18 Years." Centers for Disease Control and Prevention. July 27, 2023. https://doi.org/https://data.cdc.gov/500-Cities-Places/500-Cities-Sleeping-less-than-7-hours-among-adults/eqbn-8mpz/about_data.

CDC. "Sleep and Sleep Disorders." Centers for Disease Control and Prevention. September 12, 2022. https://www.cdc.gov/sleep/data_statistics.html.

Google "Google Maps Locations for Amazon-Arizona." Retrieved January 20, 2024, from https://maps.app.goo.gl/9z8X6zkFNXUnBiGc7

Google "Google Maps Locations for Dodger Stadium" Retrieved January 20, 2024, from https://maps.app.goo.gl/9z8X6zkFNXUnBiGc7

Google "Google Maps Locations for Empire State Building" Retrieved January 20, 2024, from https://maps.app.goo.gl/9z8X6zkFNXUnBiGc7

Google "Google Maps Locations for Houston City Hall." Retrieved January 20, 2024, from https://maps.app.goo.gl/arGxQvGk8nFPoZNa9

Google "Google Maps Locations for Mercedes-Benz Stadium" Retrieved January 20, 2024, from https://maps.app.goo.gl/9z8X6zkFNXUnBiGc7

Google "Google Maps Locations for Rainbow Grocery Cooperative" Retrieved January 20, 2024, from https://maps.app.goo.gl/bP68KRBDpB3sgbAu5

Google "Google Maps Locations next to Chicago Riverwalk-West End" Retrieved January 20, 2024, from https://maps.app.goo.gl/WcLoABvmVXJktMit5

Hale, Lauren, Terrence D. Hill, Elliot Friedman, F. Javier Nieto, Loren W. Galvao, Corinne D. Engelman, Kristen M. Malecki, and Paul E. Peppard. "Perceived Neighborhood Quality, Sleep Quality, and Health Status: Evidence from the Survey of the Health of Wisconsin." Social Science & Medicine 79, (2013): 16-22. Accessed January 19, 2024. https://doi.org/10.1016/j.socscimed.2012.07.021.

Huang and Seto (2024), Estimates of Population Highly Annoyed from Transportation Noise in the United States: An Unfair Share of the Burden by Race and Ethnicity, Environmental Impact Assessment Review, 104, 107338.  https://doi.org/10.1016/j.eiar.2023.107338

Hunter, Jaimie C., Elizabeth P. Handing, Ramon Casanova, Maragatha Kuchibhatla, Michael W. Lutz, Santiago Saldana, Brenda L. Plassman, and Kathleen M. Hayden. "Neighborhoods, Sleep Quality, and Cognitive Decline: Does where You Live and how Well You Sleep Matter?" Alzheimer's & Dementia 14, no. 4 (2018): 454-461. Accessed January 19, 2024. https://doi.org/10.1016/j.jalz.2017.10.007.

Jalali, Rostam, Habibollah Khazaei, Behnam Khaledi Paveh, Zinab Hayrani, and Lida Menati. "The Effect of Sleep Quality on Students’ Academic Achievement." Adv Med Educ Pract 11, (2020): 497-502. Accessed January 19, 2024. https://doi.org/10.2147/AMEP.S261525.
 
Nayak, Chetan S., and Araamparambil C. Anikumar. "EEG Normal Waveforms." StatPearls Publishing, (2023). Accessed January 19, 2024. https://www.ncbi.nlm.nih.gov/books/NBK539805/.

Patel, Aakash K., Vasi Reddy, Karlie R. Shumway, and John F. Araujo. "Physiology, Sleep Stages." StatPearls Publishing, (2022). Accessed January 19, 2024. https://www.ncbi.nlm.nih.gov/books/NBK526132/.

Peng, Jiaxi, Jiaxi Zhang, Bingbing Wang, Yanchen He, Qiuying Lin, Peng Fang, and Shengjun Wu. "The Relationship between Sleep Quality and Occupational Well-being in Employees: The Mediating Role of Occupational Self-efficacy." Front Psychol 14, no. 1071232 (2023). Accessed January 19, 2024. https://doi.org/10.3389/fpsyg.2023.1071232.

US Census. "American Community Survey Data." United States Census Bureau. August 16, 2023. https://www.census.gov/programs-surveys/acs/data.html.

US Census. "TIGER/Line Shapefiles." United States Census Bureau. January 9, 2024. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

US Census. "Understanding Geographic Identifiers (GEOIDs)." United States Census Bureau. October 8, 2021. https://doi.org/https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html.

Wikipedia. 2010. "Electrode Locations of International 10-20 System for EEG (Electroencephalography) Recording." Wikimedia Foundation. Last modified May 30, 2010. https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/21_electrodes_of_International_10-20_system_for_EEG.svg/1200px-21_electrodes_of_International_10-20_system_for_EEG.svg.png.

}|>
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








def select_action(state, var_name, value):
    """
    changes current raw dataset to the uploaded file
    """
    state.raw_data = state.path
    state.raw_dataset_graph = load_raw_files(state.raw_data)

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

    state.model_data = y_to_model_data(state.data)

def button_action_map(state, id):
    """
    When a map button is pressed, the above image is updated depending on the specific button pressed
    """
    assets= {
        'USA': "/images/maps/USA.jpeg",
        'HOU': "/images/maps/HOU.jpeg",
        'SEA': "/images/maps/SEA.jpeg",
        'NYC': "/images/maps/NYC.jpeg",
        'CHI': "/images/maps/CHI.jpeg",
        'ATL': "/images/maps/ATL.jpeg",
        'LAX': "/images/maps/LAX.jpeg",
        'SFO': "/images/maps/SFO.jpeg",
    }
    state.img_map = assets[id] # changes the dataset to reflect dates



def button_action_preproc(state, id):
    """
    When a map button is pressed, the above image is updated depending on the specific button pressed
    """
    assets= {
        'diagram': "images/figures/block_diagram.png",
        'eeg': "images/figures/EEG.png",
        'freq': "images/figures/pspec.png",
        'time': "images/figures/timed.png",
    }
    state.img_preproc = assets[id] # changes the dataset to reflect dates


def button_action_model(state, id):
    """
    When a model button is pressed, the img_model and txt_model should be updated accordingly, and
    the function used when running the model should be updated accordingly.
    """

    func = {
        'XGB': ML.call_FRQ(state.raw_data),
        'DNN': ML.call_DNN(state.raw_data),
        'CNN': ML.call_CNN(state.raw_data)
    }
    state.data = func[id] # changes the run function based on button ID

def download(state):
    """
    when pressed, downloads a csv of the modeled data form the raw inputs
    """
    state.dataset.to_csv('/download.csv')

def load_raw_files(X_path):
    """
    loads and downsamples raw x data to be displayed in graph
    """
    channels = ['EEG-Fpz-Cz', 'EEG-Pz-Oz', 'EOG-horizontal', 'Resp-oro-nasal', 'EMG-submental', 'Temp-rectal', 'Epoch']
    
    X = np.load(X_path)
    X_prime = np.reshape(X, (X.shape[1], -1))
    X_ds = X_prime[:, ::1000]
    idx = np.array([range(0,X_ds.shape[1])])
    X_fin = np.concatenate((X_ds, idx), axis=0)
    X_rot = np.rot90(X_fin)
    df = pd.DataFrame(data=X_rot,index=np.array(range(0,X_rot.shape[0])), columns=channels)
    return df

def load_data_files(y_path):
    y = np.load(y_path)
    y_vals = y.tolist()
    idx = range(0, y.shape[0])
    model_data = {
        "Epochs (30s)": idx,
        "Sleep Stage":y_vals
    }
    return model_data
def y_to_model_data(y):
    y_vals = y.tolist()
    idx = range(0, y.shape[0])
    model_data = {
        "Epochs (30s)": idx,
        "Sleep Stage":y_vals
    }
    return model_data

img_map = "/images/maps/USA.jpeg"
img_model = "/images/xgb_diagram.png"
img_preproc = "/images/figures/block_diagram.png"

txt_model = "howdy howdy howdy"
logo = "/images/joe.png"

model_data = load_data_files("data/p00_n1_NEW_y.npy")
data = load_data_files("data/p00_n1_NEW_y.npy")
raw_data = os.getcwd()+"/data/p00_n1_NEW_X.npy"
raw_dataset_graph = load_raw_files(raw_data) #dataframe


dataset = get_data("data/weather.csv")
start_date = datetime.date(2008, 12, 1)
end_date = datetime.date(2017, 6, 25)
download_data = "download.csv"
path = "data/weather.csv"
page = masthead+about+preproc+upload+results+acknowledge
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
Gui(page = page).run(dark_mode=False, title="PowerNap")