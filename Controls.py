from ipywidgets import widgets, interact, interactive
from IPython.display import display

initial_age = widgets.IntSlider(
    value=50,
    min=25,
    max=100,
    step=1,
    description='Age:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='i',
    slider_color='white'
)

run = widgets.ToggleButton(
    value=False,
    description='Run model',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Description',
    icon='check'
)