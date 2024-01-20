from taipy.gui import Gui
from math import cos, exp

page = """
#Re-take your sleep with *SleepX* (dogshit name)

A Value: <|{decay}|>

Decay slider: <br/>
<|{decay}|slider|>

MY chart:
<|{data}|chart|>
"""

def compute_something(decay):
    return [cos(i/16)*exp(-i*decay/6000) for i in range(720)]
decay = 10

def on_change(state, var_name, var_value):
    if var_name == 'decay':
        state.data = compute_something(var_value)


decay = 10

data = compute_something(decay)



Gui(page = page).run()
