import sys
import pickle
import plotly
import numpy as np
import pandas as pd
from scipy import stats
from Charts import *
import dash_bootstrap_components as dbc
from dash import dcc, html

# this function is basically getting the fig from the animate() function in Charts.py
# then it puts the fig inside a graph component and returns it as a layout of the animation page

def create_animations_layout(embedMtx, datDF, diffDF, selected_column):
    # labels = top 5 % of gain days and loss days
    labels = datDF[selected_column][99:].to_numpy()
    labels = np.array([stats.percentileofscore(labels, a, 'weak') for a in labels])
    labels = (labels > 95) * 1 + (labels < 5) * 1

    # color = % gain
    colors = datDF[selected_column][99:].to_numpy()
    colors = np.array([stats.percentileofscore(colors, a, 'weak') for a in colors])

    # this is because the embedding mtx skips the first 100 days
    dates = datDF.date[100:].dt.date.to_numpy()
    datDF = datDF[100:]

    fig = animate(embedMtx[:, 0:3], colors, labels, 5,  # this parameter changes number of days between frames
                  time=dates,
                  frame_duration=1, transition_duration=1,
                  line_width=2, mrkr_size=10, colorscale="Portland")

    animation_div = html.Div([
        dcc.Graph(id='animation_chart', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                  style=dict(height='', backgroundColor='#F5F5F5'), figure=fig
                  )], id='animation_div'
    )

    layout = html.Div([dbc.Spinner([animation_div], size="lg", color="primary", type="border", fullscreen=True)],

                    style=dict(display='flex', alignItems= 'center',justifyContent= 'center', width='100%'))

    return layout
