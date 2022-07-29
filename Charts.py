import copy
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objs as go
import time
import plotly
import os



# functions #


def create_frame(mtx, clr, lbl, dates=[], values=[], selected_column='', time=[],
                 line_width=1.5,
                 mrkr_size=4,
                 colorscale="Rainbow",
                 cmin=np.nan, cmax=np.nan):

    if len(clr) > 0:
        if np.isnan(cmin):
            cmin = min(clr)
        if np.isnan(cmax):
            cmax = max(clr)

        text_array = []
        # coalesce hover text
        if len(time) > 0:
            for i in range(mtx.shape[0]):
                try:
                    text_array.append(f"x:{np.round(mtx[i,0],4)}<br>y:{np.round(mtx[i,1],4)}<br>z:"
                                      f"{np.round(mtx[i,2],4)}<br>val:{np.round(clr[i],4)}<br>date:{time[i]}")
                except:
                    pass
        else:
            for i in range(mtx.shape[0]):
                try:
                    text_array.append(f"x:{np.round(mtx[i,0],4)}<br>y:{np.round(mtx[i,1],4)}<br>z:"
                                      f"{np.round(mtx[i,2],4)}<br>val:{np.round(clr[i],4)}")
                except:
                    pass

        # add marker to the last point
        lbl_scale = copy.copy(lbl)
        lbl_scale[-1] = 2  # max(2,np.round(mrkr_size/2,0).astype(np.int32))

    else:
        cmin = 0
        cmax = 0
        lbl_scale = []
        text_array = []

    try:
        hover_text = []
        for date, value in zip(dates, values):
            hover_text.append('{} : {}<br>Date : {}'.format(selected_column,
                                                            round(value, 2), date.strftime("%Y/%m/%d")))
    except:
        hover_text = text_array

    return go.Scatter3d(
                x=mtx[:, 0],
                y=mtx[:, 1],
                z=mtx[:, 2],
                text=hover_text,
                hoverinfo='text',
                line=dict(
                    color=clr,
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(thickness=5, ticktext=[cmin, cmax], outlinewidth=0),
                    width=line_width,
                ),
                marker=dict(
                    size=lbl_scale*7,
                    color=clr,
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                )
            )


def plot3D(mtx, clr, lbl, dates=[], values=[], selected_column='', line_width=1.5, mrkr_size=4,
           colorscale="Rainbow", scale=True):
    # if scale:
    #     clr = preprocessing.StandardScaler().fit_transform(clr.reshape(-1,1)).reshape(-1)
    clr = copy.copy(clr)
    fig = go.Figure(data=create_frame(mtx, clr, lbl, dates, values, selected_column,
                                      line_width=line_width, mrkr_size=mrkr_size,
                                      colorscale=colorscale))

    fig.update_layout(hoverlabel=dict(font_size=14))

    return fig


def get_ranges(fig, ndim=3):

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    z_mins = []
    z_maxs = []

    for trace_data in fig.data:
        x_mins.append(min(trace_data.x))
        x_maxs.append(max(trace_data.x))
        y_mins.append(min(trace_data.y))
        y_maxs.append(max(trace_data.y))
        if ndim == 3:
            z_mins.append(min(trace_data.z))
            z_maxs.append(max(trace_data.z))

    x_min = min(x_mins)
    x_max = max(x_maxs)
    y_min = min(y_mins)
    y_max = max(y_maxs)

    rslt = {'x': (x_min, x_max), 'y': (y_min, y_max)}

    if ndim == 3:
        z_min = min(z_mins)
        z_max = max(z_maxs)

        rslt['z'] = (z_min, z_max)

    return rslt


def frame_args(frame_duration, transition_duration=0):
    return {
        "frame": {"duration": frame_duration, "redraw": True},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": transition_duration, "easing": "linear"},
    }


def create_sliders(fig, time=[]):
    if len(time) == 0:
        time = [t for t in range(0, len(fig.frames))]

    return [{"pad": {"b": 10, "t": 60},
             "len": 0.9,
             "x": 0.1,
             "y": 0,

             "steps": [
                 {"args": [[f.name], frame_args(0)],
                  "label": str(time[k]),
                  "method": "animate",
                  } for k, f in enumerate(fig.frames)
             ]
             }
            ]


def create_menu(frame_duration, transition_duration):
    return [{"buttons": [
        {
            "args": [None, frame_args(frame_duration, transition_duration)],
            "label": "&#9654;",
            "method": "animate",
        },
        {
            "args": [[None], frame_args(0)],
            "label": "&#9724;",
            "method": "animate",
        }],

        "direction": "left",
        "pad": {"r": 10, "t": 70},
        "type": "buttons",
        "x": 0.1,
        "y": 0,
    }
    ]


def animate(mtx, clr, lbl, window, time=[],
            frame_duration=20,
            transition_duration=10,
            steps=0,
            line_width=1.5,
            mrkr_size=4,
            colorscale="Rainbow"):

    # standard scaling
    # clr_scale = preprocessing.StandardScaler().fit_transform(clr.reshape(-1,1)).reshape(-1)
    # clr_scale = preprocessing.MinMaxScaler().fit_transform(clr.reshape(-1,1)).reshape(-1)
    clr_scale = copy.copy(clr)

    # compute date labels
    if steps == 0:
        steps = len(lbl)

    dates = []
    if len(time) > 0:
        for i in np.unique(np.append(np.arange(0, steps, window), steps - 1)):
            try:
                dates.append(str(time[i]))
            except:
                pass

    else:
        dates = np.unique(np.append(np.arange(0, steps, window), steps-1)).astype(np.int32)
        dates = ['frame'+str(i) for i in time]

    # Frames
    fig = go.Figure(data=create_frame(np.full((0, 3), 0), [], []))
    cmin = min(clr_scale)
    cmax = max(clr_scale)

    frames = []
    for n, k in enumerate(np.unique(np.append(np.arange(0, steps, window), steps - 1))):
        try:
            frames.append(go.Frame(data=[create_frame(mtx[0:k+1, 0:3],
                                         clr_scale[0:k+1],
                                         lbl[0:k+1],
                                         time=time[0:k+1],
                                         line_width=line_width,
                                         mrkr_size=mrkr_size,
                                         colorscale=colorscale,
                                         cmin=cmin, cmax=cmax)],
                                   traces=[0],
                                   name=dates[n]))

        except:
            pass

    fig.update(frames=frames)

    # Scene
    fig.update_layout(
        updatemenus=create_menu(frame_duration, transition_duration),
        sliders=create_sliders(fig, dates)
    )

    # axes
    ranges = get_ranges(plot3D(mtx, clr, lbl, scale=False))
    fig.update_layout(scene=dict(xaxis=dict(range=[ranges['x'][0], ranges['x'][1]], autorange=False),
                                 yaxis=dict(range=[ranges['y'][0], ranges['y'][1]], autorange=False),
                                 zaxis=dict(range=[ranges['z'][0], ranges['z'][1]], autorange=False),
                                 aspectratio=dict(x=1, y=1, z=1)
                                 ),
                      width=1600, height=900, margin=dict(t=40, l=0, r=0, b=0))

    fig.update_layout(sliders=create_sliders(fig))

    return fig




# this function creates line fig
# this function has 2 cases
# 1- the show_ball flag is False (default) so it creates a line plot without a blue ball
# 2- the show_ball flag is True so it creates a line plot with keeping the previous blue ball

def create_line_plot(df, years_range, selected_column,
                     ball_x=None, ball_y=None, show_ball=False):

    # # # Function Parameters # # #
    # datDf: original dataframe from the pickle file
    # years_range: selected years range from slider range
    # selected_column: selected column from dropdown
    # ball_x: the x position of 3d plot blue ball ( in case it previously exists )
    # ball_y: the y position of 3d plot blue ball ( in case it previously exists )
    # show_ball: a flag that indicates whether there is a blue ball previously exist or not

    # filtering
    df.reset_index(inplace=True,drop=True)

    # creating the hover info column by looping through datDF and getting the date and value of selected column
    hov_text = []
    for ind in df.index:
        hov_text.append(
            '{} : {}<br>Date : {}'.format(selected_column,
                                          round(df[selected_column][ind], 2), df['date'][ind].strftime("%Y/%m/%d"),
                                          ))

    df['hover'] = hov_text

    # x and y of the line fig
    y = df[selected_column]
    x = df['date']

    fig = go.Figure(go.Scatter(x=x, y=y, marker_color='#1500FF', hoverinfo='text', text=df['hover']))

    # if there was an existing blue ball
    if show_ball:
        # get its hover text from ball_x and ball_y arguments
        hover_text = ['{} : {}<br>Date : {}'.format(selected_column,
                                                    round(ball_y[0], 2), ball_x[0].strftime("%Y/%m/%d"),
                                                    )]
        # add the blue ball to the fig
        fig.add_trace(go.Scatter(x=ball_x, y=ball_y, hoverinfo='text', text=hover_text,
                      mode='markers', marker=dict(size=30, opacity=0.5),
                                 ))

    fig.update_layout(xaxis_title='<b>Date<b>', yaxis_title='<b>{}<b>'.format(selected_column),
                      font=dict(size=14, family='Arial', color='black'), hoverlabel=dict(
                      font_size=18, font_family="Rockwell", font_color='white'),
                      plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(rangeslider_visible=False), margin=dict(l=0, r=0, t=20, b=0)
                      )

    fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='black')
    fig.update_layout(showlegend=False)

    return fig


# this function adds blue ball to line plot when click event is fired
def add_annotation(df, years_range, selected_column, data, annotate):

    # # # Function Parameters # # #
    # df: filtered dataframe from the callback
    # years_range: selected years range from slider range
    # selected_column: selected column from dropdown
    # data: data got from the click event
    # annotate: a flag that indicates if adding annotation text is required or not


    # create the line fig
    line_fig = create_line_plot(df, years_range, selected_column)
    # filtering
    df.reset_index(inplace=True,drop=True)

    # get blue ball data
    selected_point = data['points'][0]

    # get x and y of blue ball
    x = [df['date'][selected_point['pointNumber']]]
    y = [df[selected_column][selected_point['pointNumber']]]

    # get hover text of blue ball
    hov_text = ['{} : {}<br>Date : {}'.format(selected_column,
                                              round(y[0], 2), x[0].strftime("%Y/%m/%d"),
                                              )]

    # add the blue ball to the trace
    line_fig.add_trace(
        go.Scatter(x=x, y=y, hoverinfo='text', text=hov_text,
                   mode='markers', marker=dict(size=30, opacity=0.5),
                   ))

    # add annotation text in case the annotate flag is true
    if annotate:
        line_fig.add_annotation(
                                xref='x',
                                yref='y',
                                arrowhead=2,
                                ax=0,
                                ay=-100,
                                x=df['date'][selected_point['pointNumber']],
                                y=df[selected_column][selected_point['pointNumber']],
                                text="3-D scatter plot clicked point"
                                )

    line_fig.update_layout(showlegend=False)

    return line_fig


# this function adds orange ball to 3d fig
# this function has 2 cases
# 1- the prev_ball flag is False (default) so it adds a new orange ball to the 3d plot
# 2- the prev_ball flag is True so it keeps the previous orange ball on the newly created 3d plot
def add_annotation_3d(fig3d, data, click_3d, filtered_matrix, df, years_range, selected_column,
                      ball_x=None, ball_y=None, ball_z=None, prev_ball=False, date=None):

    # # # Function Parameters # # #
    # fig3d: the 3d plot created fig object
    # data: data got from the click event
    # click_3d: a flag indicates that the click event was from the 3d plot
    # embedMtx: original embedMtx dataframe from the pickle file
    # df: filtered dataframe from the callback
    # years_range: selected years range from slider range
    # selected_column: selected column from dropdown
    # ball_x: the x position of 3d plot blue ball ( in case it previously exists )
    # ball_y: the y position of 3d plot blue ball ( in case it previously exists )
    # ball_z: the z position of 3d plot blue ball ( in case it previously exists )
    # prev_ball: a flag that indicates whether there is an orange ball previously exist or not
    # date: the date corresponding to the previously existed orange ball


    # if there was an orange ball then add it to the newly created 3d fig
    if prev_ball:
        # getting the corresponding column value to the orange ball date
        value = df[df['date'] == date][selected_column].values[0]
        # creating the hover info of the orange ball
        hov_text = ['{} : {}<br>Date : {}'.format(selected_column,
                                                  round(value, 2), date.strftime("%Y/%m/%d"),
                                                  )]

        # add the orange ball (that was existing) to the newly created 3d fig
        fig3d.add_trace(go.Scatter3d(
            x=[ball_x],
            y=[ball_y],
            z=[ball_z], hoverinfo='text', text=hov_text,
            mode='markers',
            marker=dict(
                size=20,
                opacity=0.5, color='#00bfff'
            )
        ))

        fig3d.update_layout(showlegend=False)
        return fig3d

    # in case there was no orange ball then create a new one from the clicked data
    selected_point = data['points'][0]

    # checks if the clicked data is from the 3d plot then just directly get x, y, z from it
    if click_3d:
        x = selected_point['x']
        y = selected_point['y']
        z = selected_point['z']

    # if clicked data from line plots then get the clicked point position from corresponding point number
    else:
        x = filtered_matrix[:, 0][selected_point['pointNumber']]
        y = filtered_matrix[:, 1][selected_point['pointNumber']]
        z = filtered_matrix[:, 2][selected_point['pointNumber']]

    # getting the hover info from the corresponding point number
    date = df.reset_index()['date'][selected_point['pointNumber']]
    value = df.reset_index()[selected_column][selected_point['pointNumber']]

    hov_text = ['{} : {}<br>Date : {}'.format(selected_column,
                                              round(value, 2), date.strftime("%Y/%m/%d"),
                                              )]

    # adding the orange ball to the 3d plot
    fig3d.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z], hoverinfo='text', text=hov_text,
                    mode='markers',
                    marker=dict(size=20, opacity=0.5, color='#00bfff')))

    fig3d.update_layout(showlegend=False)

    return fig3d
