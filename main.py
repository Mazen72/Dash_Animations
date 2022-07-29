import dash
import pandas as pd
import numpy as np
from scipy import stats
import base64
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import io
from dash import Dash, Input, Output, dash_table, callback_context, State
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.exceptions import PreventUpdate
import pickle
from collections import OrderedDict
from Charts import *
from Animations import *
import os
from datetime import datetime

# defining server object
server = Flask(__name__)

# defining app object
app = dash.Dash(
    __name__, server=server,
    meta_tags=[
        {
            'charset': 'utf-8',
        },
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0, shrink-to-fit=no'
        }
    ],
)

# setting the title of the app which will be shown in browser tab
app.title = 'Dashboard'

# setting some callbacks error handling in the app
app.config.suppress_callback_exceptions = True

# getting the local directory of the app
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# getting the directory of the embedMtx .pickle data
embedMtx_pkl = os.path.join(THIS_FOLDER, 'embedMtx.pkl')
# getting the directory of the datDF .pickle data
datDF_pkl = os.path.join(THIS_FOLDER, 'datDF.pkl')
# getting the directory of the diffDF .pickle data
diffDF_pkl = os.path.join(THIS_FOLDER, 'diffDF.pkl')

# reading embedMtx .pickle data
pkl_in = open(embedMtx_pkl, 'rb')
embedMtx = pickle.load(pkl_in)
print(embedMtx.shape,embedMtx[:,0])
pkl_in.close()

# reading datDF .pickle data
pkl_in = open(datDF_pkl, 'rb')
datDF = pickle.load(pkl_in)
pkl_in.close()

# reading diffDF .pickle data
pkl_in = open(diffDF_pkl, 'rb')
diffDF = pickle.load(pkl_in)
pkl_in.close()

# df = datDF[100:]
# df.reset_index(inplace=True)
# df['x']=embedMtx[:,0]
# df['y']=embedMtx[:,1]
# df['z']=embedMtx[:,2]
# print(df.head(3))
# df.to_csv('dataframe.csv',index=False)

# defining a dictionery contains some colors used in some main parts of the app layout
components_colors = {'secondary': '#0b1a50', 'bg': '#e7f0f9', 'cards': 'white', 'fonts': 'black'}

# font size used in most of the app
text_font_size = '1.7vh'

# creating the dashboard title header that is dispayed on the top of the dashboard
header_text = html.Div('Dashboard Title', id='main_header_text', className='main-header',
                       style=dict(color=components_colors['cards'],
                                  fontWeight='bold', fontSize='2.2vh', width='100%', paddingTop='1vh',
                                  paddingBottom='1vh', display='flex', alignItems='center', justifyContent='center'))

# creating the title header column and setting its spacing
header_text_col = dbc.Col([header_text],
                          xs=dict(size=10, offset=1), sm=dict(size=10, offset=1),
                          md=dict(size=8, offset=2), lg=dict(size=6, offset=3), xl=dict(size=6, offset=3))



upload_csv_button = html.Div(dbc.Button(
    "Upload Csv", id="upload_csv_button", n_clicks=0, size='lg',className="animate_btn",
    style=dict(fontSize='1.4vh', backgroundColor='#119DFF',color='white',fontWeight='bold')
), style=dict(textAlign='center', display='', marginTop='1.5%', paddingLeft='',width=''))

upload_csv_text = html.Div(html.H1('Upload CSV File',
                              style=dict(fontSize='1.6vh', fontWeight='bold', color='black',
                                         marginTop='')),
                      style=dict(textAlign="center", width='100%'))

upload_component =dcc.Upload(id='upload_csv',children=[upload_csv_button] ,

                            style=dict()
                            )

upload_msg=dbc.Spinner(dbc.Alert("Uploaded Succesfully", id="upload_msg", is_open=False, duration=3500,
                           color="success",style=dict(marginTop='2vh',textAlign='center',fontSize='1.5vh'))
                             , size="md", color="light", type="border", fullscreen=False,
                             spinner_style=dict(marginTop=''))

upload_csv_div = html.Div([upload_csv_text,upload_component],
                          #style={'width': '100%'}
                          )

upload_csv_row=html.Div(upload_csv_div,style=dict(display= 'flex', alignItems= 'center',
                                                        justifyContent= 'center',width='100%'))

upload_csv_col=dbc.Col( dbc.Card(dbc.CardBody([upload_csv_row,upload_msg
                                                          ])
                                            , style=dict(backgroundColor='white', width='',
                                            border=''), id='csv_card',
                                            className='shadow my-2 mx-3 css class')
,        xs=dict(size=12,offset=0), sm=dict(size=12,offset=0),
        md=dict(size=8,offset=2), lg=dict(size=2,offset=1), xl=dict(size=2,offset=1) ,
                            style=dict( paddingTop='1vh'))


def get_layout_columns(datDF):

    # creating new column in datDf called year which extract the year from the corresponding date
    # the goal from it is to be used in the range slider component
    datDF['Year'] = pd.DatetimeIndex(datDF['date']).year.astype(str)
    # converting years column to int type
    datDF['Year'] = datDF['Year'].astype('int32')
    # converting years column to list
    years = datDF['Year'].to_list()
    # removing repeated years from years list
    years = list(OrderedDict.fromkeys(years))

    # setting the marks values in the range slider which will be all years from years list
    marks_values = {year: {'label': '{}'.format(year), 'style': {'color': 'black'}} for year in years}

    # creating the text displayed on top of the range slider
    slider_text = html.Div(html.H1('Select Years Range',
                                   style=dict(fontSize=text_font_size, fontWeight='bold', color='black', )),
                           style=dict(textAlign="center", width='100%'))

    # creating the range slider component
    years_slider = html.Div([dcc.RangeSlider(min=years[0], max=years[-1], step=1, value=[years[0], years[-1]],
                                             marks=marks_values, id='years_slider')
                             ])

    # creating a div that contains both the text and range slider
    years_slider_div = html.Div([slider_text, years_slider])

    # creating the column that will contain the above div within a card component with setting the spacing
    slider_card=html.Div(dbc.Card(dbc.CardBody([years_slider_div]),
                                              style=dict(backgroundColor='white', width='100%', ), id='slider_card',
                                              className='shadow my-2 mx-3 css class'),
                                     style=dict(display='flex', alignItems='center',
                                                justifyContent='center', width='100%'))

    slider_column = dbc.Col(slider_card,
                            xs=dict(size=12, offset=0), sm=dict(size=12, offset=0),
                            md=dict(size=8, offset=2), lg=dict(size=6, offset=0), xl=dict(size=6, offset=0),
                            style=dict(paddingTop='1vh'),id='slider_column')

    # chart1 is the 3d plot
    # creating chart1 title text
    chart1_title = html.Div(html.H1('After Wavelet-UMAP transformation',
                                    style=dict(fontSize=text_font_size, fontWeight='bold', color='black')),
                            style=dict(textAlign="center", width='100%'))

    # creating an empty plot so that the graphs components are initiallized with it at the beginning
    # and that is before the callback are fired after the app starts
    empty_fig = go.Figure()
    empty_fig.update_layout(font=dict(size=14, family='Arial', color='black'),
                            hoverlabel=dict(font_size=16, font_family="Rockwell",
                                            font_color='white', bgcolor=components_colors['secondary']),
                            plot_bgcolor='white',
                            paper_bgcolor='white', xaxis=dict(rangeslider_visible=False),
                            margin=dict(l=0, r=0, t=20, b=0))

    empty_fig.update_xaxes(showgrid=False, showline=True, zeroline=False, linecolor='black')
    empty_fig.update_yaxes(showgrid=False, showline=True, zeroline=False, linecolor='black')

    # creating chart1 graph component
    chart1_div = html.Div([
        dcc.Graph(id='chart1', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                  style=dict(height='60vh', backgroundColor='#F5F5F5'), figure=empty_fig
                  )], id='chart1_div')

    # dropdown1 is the dropdown menu of the 3d plot
    # creating the options list of the dropdown ( which will be same for all dropdowns
    # the list will be all columns of datDf except the date and Year columns
    dropdown1_options = list(datDF.columns)
    dropdown1_options.remove('date')
    dropdown1_options.remove('Year')

    # creating chart1 dropdown menu
    dropdown1 = dcc.Dropdown(id='dropdown1', options=[{'label': col, 'value': col} for col in dropdown1_options],
                             value=dropdown1_options[0],
                             style=dict(color='white', fontWeight='bold', textAlign='center', borderRadius='5%',
                                        width='100%', backgroundColor='#0b1a50', border='1px solid #0b1a50'))

    # creating a div that contains chart1 dropdown menu
    dropdown_div1 = html.Div([dropdown1],
                             style=dict(fontSize=text_font_size, width='20%', display='inline-block'))

    # creating the Animate button and making it open a new tab upon clicking it
    # this new tab will be updated with the animation layout from callbacks
    button_div1 = html.Div(dbc.Button("Animate", id="animate_button", className="animate_btn", n_clicks=0, size='lg',
                                      href='/Animations.html', target='_blank',
                                      style=dict(fontSize=text_font_size, backgroundColor='#119DFF'),
                                      external_link=True),
                           style=dict(textAlign='center', display='inline-block', marginTop='', paddingLeft='2vw'))

    # creating a div that contains both the dropdown and animate button and places them to the center
    options_div1 = html.Div([dropdown_div1, button_div1],
                            style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                   'justify-content': 'center'})

    # creating the column that will contain the above div within a card component with setting the spacing
    chart1_card=dbc.Card([dbc.CardHeader(dbc.Row([chart1_title]), style=dict(backgroundColor='white')),
                                   dbc.CardBody([dbc.Spinner([chart1_div],
                                                             size="lg", color="primary", type="border",
                                                             fullscreen=False),
                                                 html.Br(), options_div1])],
                                  style=dict(backgroundColor=components_colors['cards'], id='chart1_card'), id='col1',
                                  className='shadow my-2 mx-3 css class')

    chart1_col = dbc.Col(chart1_card,
                         xs=dict(size=12, offset=0), sm=dict(size=12, offset=0),
                         md=dict(size=12, offset=0), lg=dict(size=6, offset=0), xl=dict(size=6, offset=0),
                         style=dict(paddingRight='0.5vw', paddingLeft='0.5vw', paddingTop=''),id='chart1_col')

    # dropdown2 is the dropdown menu of the upper line plot
    dropdown2 = dcc.Dropdown(id='dropdown2', options=[{'label': col, 'value': col} for col in dropdown1_options],
                             value=dropdown1_options[0],
                             style=dict(color='white', fontWeight='bold', textAlign='center', borderRadius='5%',
                                        width='100%', backgroundColor='#0b1a50', border='1px solid #0b1a50'))

    # creating a div that contains the dropdown menu
    dropdown_div2 = html.Div([dropdown2],
                             style=dict(fontSize=text_font_size, width='20%', display='inline-block'))

    # creating a div that contains the dropdown and places it to the center
    options_div2 = html.Div([dropdown_div2],
                            style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                   'justify-content': 'center'})

    # chart2 is the upper line plot
    # creating chart2 graph component
    chart2_div = html.Div(
        [dcc.Graph(id='chart2', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                   style=dict(height='26vh', backgroundColor='#F5F5F5'), figure=empty_fig)],
        id='chart2_div')

    # creating a card that contains both the upper line plot div and its dropdown div
    upper_card = dbc.Card([dbc.CardBody([dbc.Spinner([chart2_div],
                                                     size="lg", color="primary", type="border", fullscreen=False),
                                         html.Br(), options_div2])],
                          style=dict(backgroundColor=components_colors['cards']), id='col2',
                          className='shadow my-2 mx-3 css class')

    # dropdown3 is the dropdown menu of the lower line plot
    dropdown3 = dcc.Dropdown(id='dropdown3', options=[{'label': col, 'value': col} for col in dropdown1_options],
                             value=dropdown1_options[1],
                             style=dict(color='white', fontWeight='bold', textAlign='center', borderRadius='5%',
                                        width='100%', backgroundColor='#0b1a50', border='1px solid #0b1a50'))

    # creating a div that contains the dropdown menu
    dropdown_div3 = html.Div([dropdown3],
                             style=dict(fontSize=text_font_size, width='20%', display='inline-block'))

    # creating a div that contains the dropdown and places it to the center
    options_div3 = html.Div([dropdown_div3], style={'width': '100%', 'display': 'flex', 'align-items': 'center',
                                                    'justify-content': 'center'})

    # chart3 is the lower line plot
    # creating chart3 graph component
    chart3_div = html.Div(
        [dcc.Graph(id='chart3', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                   style=dict(height='26vh', backgroundColor='#F5F5F5'), figure=empty_fig)],
        id='chart3_div')

    # creating a card that contains both the lower line plot div and its dropdown div
    lower_card = dbc.Card([dbc.CardBody([dbc.Spinner([chart3_div],
                                                     size="lg", color="primary", type="border", fullscreen=False),
                                         html.Br(), options_div3])],
                          style=dict(backgroundColor=components_colors['cards'], paddingTop=''), id='col3',
                          className='shadow my-2 mx-3 css class')

    # creating a column that contains both the upper and lower line plots cards and setting the spacing
    line_plots_cards=[upper_card, lower_card]
    line_plots_col = dbc.Col(line_plots_cards,
                             xs=dict(size=12, offset=0), sm=dict(size=12, offset=0),
                             md=dict(size=12, offset=0), lg=dict(size=6, offset=0), xl=dict(size=6, offset=0),
                             style=dict(paddingRight='0.5vw', paddingLeft='0.5vw', paddingTop=''),id='line_plots_col')


    return (slider_card, chart1_card, line_plots_cards)


slider_card, chart1_card, line_plots_cards=get_layout_columns(datDF)
slider_column = dbc.Col(slider_card,
                        xs=dict(size=12, offset=0), sm=dict(size=12, offset=0),
                        md=dict(size=8, offset=2), lg=dict(size=6, offset=0), xl=dict(size=6, offset=0),
                        style=dict(paddingTop='1vh'), id='slider_column')

chart1_col = dbc.Col(chart1_card,
                     xs=dict(size=12, offset=0), sm=dict(size=12, offset=0),
                     md=dict(size=12, offset=0), lg=dict(size=6, offset=0), xl=dict(size=6, offset=0),
                     style=dict(paddingRight='0.5vw', paddingLeft='0.5vw', paddingTop=''), id='chart1_col')

line_plots_col = dbc.Col(line_plots_cards,
                         xs=dict(size=12, offset=0), sm=dict(size=12, offset=0),
                         md=dict(size=12, offset=0), lg=dict(size=6, offset=0), xl=dict(size=6, offset=0),
                         style=dict(paddingRight='0.5vw', paddingLeft='0.5vw', paddingTop=''), id='line_plots_col')
# creating the main layout from the previous created columns
main_layout = html.Div([dbc.Row([header_text_col],
                                style=dict(backgroundColor=components_colors['secondary']), id='main_header'),
                        dbc.Row([upload_csv_col, slider_column]), html.Br(), dbc.Row([chart1_col, line_plots_col]),
                            ],id='main_layout')

# adding the main layout variable to the app layout object
# also adding dcc.Location() component which is used in url operations handling ( when animate button pressed )
app.layout = html.Div([html.Div(children=main_layout, id='layout'),
                       dcc.Location(id='url', refresh=True),
                       dcc.Store(id='uploaded_df', data=None, storage_type='memory'),
                       html.Br(), html.Br(), html.Br()],
                      style=dict(backgroundColor=components_colors['bg']), className='main')

# # # # # Callbacks # # # # #

# # # uploading the dataframe # # #
# States
# upload_csv filename: the uploaded file name got from upload_csv component when uploading a file

# Inputs
# upload_csv contents: the uploaded file content got from upload_csv component when uploading a file

# Outputs
# upload_msg is_open: opens the alert component which shows a massage of the upload status
# upload_msg children: the text shown on the alert component
# upload_msg color: the color of the alert component
# uploaded_df data: storing the dataframe upload on a dcc.Store() component called uploaded_df
# slider_column children: updating the slider column layout with the uploaded dataframe data
# chart1_col children: updating the 3d plot column layout with the uploaded dataframe data
# line_plots_col children: updating the 2 line plots column layout with the uploaded dataframe data

@app.callback([Output("upload_msg", "is_open"),Output("upload_msg", "children"),
               Output("upload_msg", "color"),Output('uploaded_df', 'data'),Output('slider_column','children'),
               Output('chart1_col','children'),Output('line_plots_col','children')],
               Input('upload_csv', 'contents'),
               State('upload_csv', 'filename'),

            prevent_initial_call=True)
def upload_dataframe(list_of_contents,filename):
    # checking if the upload component content is not None (an upload process occured) else exist callback without updating
    if list_of_contents is not None:
        # checking if the file format end with .csv else raise an error to user
        if '.csv' in filename:
            # parsing the upload component content
            content_type, content_string = list_of_contents.split(',')
            decoded = base64.b64decode(content_string)

            # reading the uploaded csv and raising sn error to user in case reading failed (wrong csv content)
            try:
                uploaded_df=pd.read_csv(io.StringIO(decoded.decode('utf-8')), parse_dates=True, infer_datetime_format=True)

            except:
                return (True, 'Upload Error: csv file not formatted correctly', 'danger', dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update)

            uploaded_df=uploaded_df.reset_index(drop=True)

            print(uploaded_df.head(3))

            # checking that 'date','x','y','z' columns exist in csv else raise an error to user
            try:
                test_df=uploaded_df[['date','x','y','z']]
            except:
                return (True, 'Upload Error: wrong columns names', 'danger', dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update)

            # getting the new columns content after uploading the dataframe
            slider_card, chart1_card, line_plots_cards = get_layout_columns(uploaded_df.drop(['x', 'y', 'z'],axis=1))



            return (True, 'Uploaded Succesfully', 'success', uploaded_df.to_dict('records'),
                    slider_card, chart1_card, line_plots_cards)

        else:
            return (True, 'Upload Error: file format must be .csv', 'danger', dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update)

    else:
        raise PreventUpdate


# # # updating the page # # #
# States
# dropdown1 value: the current selected value from the 3d plot dropdown

# Inputs
# url pathname: when the page url path changes

# Outputs
# layout children: the layout of the page

@app.callback(Output('layout', 'children'), Input('url', 'pathname'), State('dropdown1', 'value'))
def update_page(pathname, selected_column):
    # if the url ends with '/' (the default) return the main layout
    if pathname == '/':
        return main_layout

    # if the url ends with '/Animations.html' (animate button clicked) return the animation layout
    # the animation layout has a timer that fires callback of the animate() functions in charts.py after 0.5 second
    # the reason why i didnt call the function directly is related to preventing backend synchronization issue
    elif pathname == '/Animations.html':
        main_div = html.Div(html.H1('Loading Animations...',
                            style=dict(fontSize=text_font_size, fontWeight='bold', color='black')),
                            id='animation_main_div',
                            style=dict(textAlign="center", width='100%'))

        return [html.Br(), dbc.Spinner([main_div], size="lg", color="primary", type="border", fullscreen=True),
                dcc.Interval(interval=500, n_intervals=0, id='timer', max_intervals=1),
                dcc.Store(id='selected_column', data=selected_column)]

    else:
        return ''


# # # updating the animated chart # # #
# States
# selected_column value: the current selected value from the 3d plot dropdown (got from previous callback)

# Inputs
# timer passes 0.5 second: after the animate button pressed this callback called after 0.5 second

# Outputs
# animation chart div: the graph component of the animation chart to be updated from animate() function

@app.callback(Output('animation_main_div', 'children'), Input('timer', 'n_intervals'), State('selected_column', 'data'))
def update_animated_chart(timer, selected_column):

    # getting the animation layout from create_animations_layout() function in Animation.py
    # this function basically calls the animate() function in Charts.py inside it
    animations_layout = create_animations_layout(embedMtx, datDF, diffDF, selected_column)

    return animations_layout


# # # updating the 3d plot # # #
# States
# chart1 figure: the current state of the 3d plot figure
# uploaded_df: the current uploaded dataframe value if exist

# Inputs
# dropdown1 value: selected value of the dropdown
# years_slider value: selected value of years range
# chart1 clickData : click event of 3d plot
# chart2 clickData : click event of upper line plot
# chart3 clickData : click event of lower line plot

# Outputs
# chart1 figure: the 3d plot figure

@app.callback(Output('chart1', 'figure'),
              [Input('dropdown1', 'value'), Input('years_slider', 'value'), Input('chart1', 'clickData'),
               Input('chart2', 'clickData'), Input('chart3', 'clickData')],
              [State('chart1', 'figure'),State('uploaded_df', 'data')])
def update_3d_plot(selected_column, years_range, data, line1_data, line2_data, fig1,uploaded_df):
    # getting the clicked input id
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # checking if there is a dataframe uploaded or the default one is being used and setting the dataframe value depending on it
    if uploaded_df == None:
        df = datDF[100:]
        df.reset_index(inplace=True)
        df = df[(df['date'].dt.year >= years_range[0]) & (df['date'].dt.year <= years_range[1])]
        filtered_matrix=embedMtx[df.index[0]:df.index[-1], 0:3]

    else:
        df=pd.DataFrame(uploaded_df)
        df['date']=pd.to_datetime(df['date'])
        df = df[(df['date'].dt.year >= years_range[0]) & (df['date'].dt.year <= years_range[1])]
        filtered_matrix=df[['x', 'y', 'z']].to_numpy()

    # labels = top 5 % of gain days and loss days
    labels = df[selected_column].to_numpy()
    labels = np.array([stats.percentileofscore(labels, a, 'weak') for a in labels])
    labels = (labels > 95) * 1 + (labels < 5) * 1

    # color = % gain
    colors = df[selected_column].to_numpy()
    colors = np.array([stats.percentileofscore(colors, a, 'weak') for a in colors])

    # creating lists of selected dates and corresponding values to be used in plot3d() function in charts.py
    # they are used in creating the hover text column that will display the hover info in created fig
    dates = df.reset_index()['date']
    values = df.reset_index()[selected_column]

    # creating the 3d plot fig
    fig = plot3D(filtered_matrix, colors, labels, dates, values, selected_column)

    # the part bellow is basically for the case where dropdown or years slider changes
    # while there is an orange ball on 3d plot
    # in this case we want to keep the orange ball on the newly created fig

    # if the number of traces inside the fig > 1 ( there is an orange ball )
    if len(fig1['data']) > 1:
        # get the x, y, z of the orange ball
        x = fig1['data'][1]['x']
        y = fig1['data'][1]['y']
        z = fig1['data'][1]['z']
        # get the hover info of the orange ball
        hov_text = fig1['data'][1]['text']
        # get the date from hover info
        date = hov_text[0].split(':')[2][1:]
        x_date = pd.to_datetime(date, format="%Y-%m-%d")
        # extract the year
        x_year = x_date.year

        # if the input is years slider or dropdown and the orange ball year is within the slider range
        if (input_id == 'years_slider' or input_id == 'dropdown1') and (x_year >= years_range[0])\
                and (x_year <= years_range[1]):

            # x, y, z of the orange ball
            ball_x = x[0]
            ball_y = y[0]
            ball_z = z[0]
            # add the orange ball to the newly created fig from the add_annotation_3d function in Charts.py
            return add_annotation_3d(fig, None, None, filtered_matrix, df, years_range, selected_column,
                                     ball_x, ball_y, ball_z, True, x_date)

        else:
            pass

    else:
        pass

    # in case the previous check is false ( 3d plot has no orange ball )
    # if the input is not a click event so the input is either dropdown or years slider
    # then return the newly created fig
    if input_id != 'chart1' and input_id != 'chart2' and input_id != 'chart3':
        return fig

    # if input is 3d plot click event
    # add the orange ball to the newly created fig from the add_annotation_3d function in Charts.py
    elif input_id == 'chart1':
        fig = add_annotation_3d(fig, data, True, filtered_matrix, df, years_range, selected_column)
        return fig

    # if input is upper line plot click event
    # add the orange ball to the newly created fig from the add_annotation_3d function in Charts.py
    elif input_id == 'chart2':
        fig = add_annotation_3d(fig, line1_data, False, filtered_matrix, df, years_range, selected_column)
        return fig

    # if input is lower line plot click event
    # add the orange ball to the newly created fig from the add_annotation_3d function in Charts.py
    elif input_id == 'chart3':
        fig = add_annotation_3d(fig, line2_data, False, filtered_matrix, df, years_range, selected_column)
        return fig

# # # updating the upper line plot # # #
# States
# chart1 figure: the current state of the upper line plot
# uploaded_df: the current uploaded dataframe value if exist

# Inputs
# dropdown2 value: selected value of the dropdown
# years_slider value: selected value of years range
# chart1 clickData : click event of 3d plot
# chart2 clickData : click event of upper line plot
# chart3 clickData : click event of lower line plot

# Outputs
# chart1 figure: the upper line plot figure
@app.callback(Output('chart2', 'figure'),
              [Input('dropdown2', 'value'), Input('years_slider', 'value'), Input('chart1', 'clickData'),
               Input('chart2', 'clickData'), Input('chart3', 'clickData')],
              [State('chart2', 'figure'),State('uploaded_df', 'data')]
              )
def update_line_plot1(selected_column, years_range, data, line1_data, line2_data, fig2,uploaded_df):

    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if uploaded_df == None:
        # filtering datDf based on selected year range with removing first 100 rows
        df = datDF[100:]
        df.reset_index(inplace=True, drop=True)
        df = df[(df['date'].dt.year >= years_range[0]) & (df['date'].dt.year <= years_range[1])]

    else:
        df=pd.DataFrame(uploaded_df)
        df['date']=pd.to_datetime(df['date'])
        df = df[(df['date'].dt.year >= years_range[0]) & (df['date'].dt.year <= years_range[1])]

    # the part bellow is basically for the case where dropdown or years slider changes
    # while there is an blue ball on line plot
    # in this case we want to keep the blue ball on the newly created fig

    # if the number of traces inside the fig > 1 ( there is an blue ball )
    if len(fig2['data']) > 1:
        # get the x (date) of the blue ball
        x = fig2['data'][1]['x'][0]
        x_date = pd.to_datetime(x[:10], format="%Y-%m-%d")
        x_year = x_date.year

        # if the input is years slider or dropdown and the blue ball year is within the slider range
        if (input_id == 'years_slider' or input_id == 'dropdown2') and (x_year >= years_range[0]) and\
                (x_year <= years_range[1]):
            ball_x = [x_date]
            # getting the corresponding column value from the date
            ball_y = df[df['date'] == x_date][selected_column].to_list()
            # returning the line plot from create_line_plot function in charts.py
            # with setting the show_ball flag to true to keep the previous existed ball when fig updates
            return create_line_plot(df, years_range, selected_column, ball_x=ball_x, ball_y=ball_y, show_ball=True)

        else:
            pass

    else:
        pass

    # in case the previous check is false ( line plot has no blue ball )
    # if the input is not a click event so the input is either dropdown or years slider
    # then return the newly created fig
    if input_id != 'chart1' and input_id != 'chart2' and input_id != 'chart3':
        return create_line_plot(df, years_range, selected_column)

    # if input is 3d plot click event
    # add the orange ball to the newly created fig from the add_annotation function in Charts.py
    elif input_id == 'chart1':
        return add_annotation(df, years_range, selected_column, data, True)

    # if input is upper line plot click event
    # add the orange ball to the newly created fig from the add_annotation function in Charts.py
    elif input_id == 'chart2':

        return add_annotation(df, years_range, selected_column, line1_data, False)

    # if input is lower line plot click event
    # add the orange ball to the newly created fig from the add_annotation function in Charts.py
    elif input_id == 'chart3':

        return add_annotation(df, years_range, selected_column, line2_data, False)


# # # SAME AS THE FUNCTION ABOVE BUT WITH LOWER LINE PLOT # # #

@app.callback(Output('chart3', 'figure'),
              [Input('dropdown3', 'value'), Input('years_slider', 'value'), Input('chart1', 'clickData'),
               Input('chart3', 'clickData'), Input('chart2', 'clickData')],
              [State('chart3', 'figure'), State('uploaded_df', 'data')])
def update_line_plot2(selected_column, years_range, data, line2_data, line1_data, fig3, uploaded_df):

    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if uploaded_df == None:
        # filtering datDf based on selected year range with removing first 100 rows
        df = datDF[100:]
        df.reset_index(inplace=True, drop=True)
        df = df[(df['date'].dt.year >= years_range[0]) & (df['date'].dt.year <= years_range[1])]

    else:
        df=pd.DataFrame(uploaded_df)
        df['date']=pd.to_datetime(df['date'])
        df = df[(df['date'].dt.year >= years_range[0]) & (df['date'].dt.year <= years_range[1])]

    if len(fig3['data']) > 1:
        x = fig3['data'][1]['x'][0]
        x_date = pd.to_datetime(x[:10], format="%Y-%m-%d")
        x_year = x_date.year

        if (input_id == 'years_slider' or input_id == 'dropdown3') and (x_year >= years_range[0]) and\
                (x_year <= years_range[1]):

            ball_x = [x_date]
            ball_y = df[df['date'] == x_date][selected_column].to_list()
            return create_line_plot(df, years_range, selected_column, ball_x=ball_x, ball_y=ball_y, show_ball=True)

        else:
            pass

    else:
        pass

    if input_id != 'chart1' and input_id != 'chart3' and input_id != 'chart2':
        return create_line_plot(df, years_range, selected_column)

    elif input_id == 'chart1':
        return add_annotation(df, years_range, selected_column, data, True)

    elif input_id == 'chart3':

        return add_annotation(df, years_range, selected_column, line2_data, False)

    elif input_id == 'chart2':

        return add_annotation(df, years_range, selected_column, line1_data, False)



if __name__ == '__main__':
    app.run_server(host='localhost', port=8050, debug=False, dev_tools_silence_routes_logging=True)
