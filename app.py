import pathlib
import os

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import constants

import dash_table
import matplotlib.pyplot as plt
import plotly.express as px
from googletrans import Translator
import plotly.offline as pyo
import plotly.graph_objs as go
import json

# Data Reads
demographics = pd.read_csv('data\demographics.csv')
employment = pd.read_csv('data\employment.csv')
mobility = pd.read_csv('data\mobility.csv')
cases = pd.read_csv(
    'https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data/rivm_NL_covid19_total_municipality.csv')
income = pd.read_csv('data\income.csv')
coordinates = pd.read_csv('data\coordinates.csv')
clustering = pd.read_csv('data\H5cluster.csv')
provinces = pd.read_excel('data\gemeente_provinces.xls')


# Translating the columns
translator = Translator()
cases.rename(columns=lambda x: translator.translate(x).text, inplace=True)

# Removing commas and changing the data types
for i in range(1, demographics.shape[1]):
    if (demographics.iloc[:, i]).dtype == 'object':
        demographics.iloc[:, i] = demographics.iloc[:, i].str.replace(',', '.')
        demographics.iloc[:, i] = demographics.iloc[:, i].astype('float')

for i in range(1, mobility.shape[1]):
    if (mobility.iloc[:, i]).dtype == 'object':
        mobility.iloc[:, i] = mobility.iloc[:, i].str.replace(',', '.')
        mobility.iloc[:, i] = mobility.iloc[:, i].astype('float')

income = income.dropna(axis=1)
for i in range(1, income.shape[1]):
    if (income.iloc[:, i]).dtype == 'object':
        income.iloc[:, i] = income.iloc[:, i].replace('?', '0')
        income.iloc[:, i] = income.iloc[:, i].str.replace(',', '.')
        income.iloc[:, i] = income.iloc[:, i].astype('float')

# Calculating the percentages
demographics.iloc[:, np.r_[4:20]] = demographics.iloc[:, np.r_[4:20]].div(demographics.iloc[:, 1], axis=0)
demographics.iloc[:, np.r_[26:28]] = demographics.iloc[:, np.r_[26:28]].div(demographics.iloc[:, 25], axis=0)
demographics.iloc[:, np.r_[24:26]] = demographics.iloc[:, np.r_[24:26]].div(demographics.iloc[:, 23], axis=0)

# Aggregating the total cases per municipality
cases = cases.dropna()
cases = cases.rename(columns={'municipality Name': 'Municipalities'})
cases = cases.replace("'s-Gravenhage", 'Den Haag')
cases = cases.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')

grp1 = cases[cases.date == max(cases.date)]
grp1 = grp1.iloc[:, np.r_[2, 5]]

# provinces
provinces = provinces.drop(columns=['Gemeentecode', 'GemeentecodeGM', 'Provinciecode', 'ProvinciecodePV'])
provinces = provinces.rename(
    columns={'Gemeentenaam': 'Municipality', 'Provincienaam': 'Provinces'})
provinces.Municipality = provinces.Municipality.str.replace("'s-Gravenhage", "s-Gravenhage")

# coordinates
coordinates = pd.merge(coordinates, provinces, on='Municipality')
coordinates = coordinates.rename(
    columns={'Municipality': 'Municipalities', 'Latitude (generated)': 'lat', 'Longitude (generated)': 'lon'})

# correct misspellings to match
coordinates.Municipalities = coordinates.Municipalities.str.replace("s-Gravenhage", 'Den Haag')
coordinates.Municipalities = coordinates.Municipalities.str.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')
coordinates = coordinates.apply(lambda x: x.str.replace(',', '.'))

# Geojson with NL data
with open('data\gemeente.geojson') as json_data:
    nl_data = json.load(json_data)

# Join data with municipality id codes (to work with the geojson)
# Read data from website
mun_codes = pd.read_excel('https://www.cbs.nl/-/media/_excel/2020/03/gemeenten-alfabetisch-2020.xlsx')

# Select columns of interest
mun_codes = mun_codes.filter(['GemeentecodeGM', 'Gemeentenaam'], axis=1)
# Rename columns
mun_codes.columns = ['id', 'Municipalities']
# Correct municipalities names
mun_codes.Municipalities = mun_codes.Municipalities.str.replace("'s-Gravenhage", 'Den Haag')
mun_codes.Municipalities = mun_codes.Municipalities.str.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')

# joining the data with the clusters
joined_data = pd.merge(grp1, mun_codes, how='left', on='Municipalities')
joined_data = pd.merge(joined_data, coordinates, how='left', on='Municipalities')
joined_data = pd.merge(joined_data, clustering, how='left', on='Municipalities')
joined_data.Cluster = joined_data.Cluster.astype(str)


# NL latitude and longitude values
latitude = 52.370216
longitude = 4.895168

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

missedData = pd.read_csv('data\Missed-data.csv')


# New case function
def newCase(num):
    num_dropna = num.dropna()
    num_dropna.reset_index(inplace=True)
    num_dropna = num_dropna[['date', 'Municipalities', 'Provincienaam', 'Number']]
    new_cases = pd.DataFrame(
        data={'date': [], 'Municipalities': [], 'Provincienaam': [], 'Number': [], 'New_cases': []})
    Municipality = num_dropna['Municipalities'].unique()
    for i in Municipality:
        M = num_dropna[num_dropna['Municipalities'] == i]
        M.sort_values(by='date', inplace=True)
        M.reset_index(inplace=True)
        M = M[['date', 'Municipalities', 'Provincienaam', 'Number']]
        row_1st = pd.DataFrame(data={'date': [M.iloc[0]['date']], 'Municipalities': [M.iloc[0]['Municipalities']],
                                     'Provincienaam': [M.iloc[0]['Provincienaam']], 'Number': [M.iloc[0]['Number']],
                                     'New_cases': [M.iloc[0]['Number']]})
        new = M['Number'].diff().tolist()
        M.drop([0], inplace=True)
        new.pop(0)
        M['New_cases'] = new
        dif = pd.concat([row_1st, M], ignore_index=True)
        new_cases = pd.concat([new_cases, dif], ignore_index=True)
    return new_cases


missedData = missedData.replace("'s-Gravenhage", 'Den Haag')
missedData = missedData.replace('Nuenen, Gerwen en Nederwetten', 'Nuenen c.a.')

# Adding the missed data to the new cases
new_cases = newCase(cases)
new_cases.set_index('date', inplace=True)
new_cases.drop('2020-04-08', inplace=True)
new_cases.reset_index(level=0, inplace=True)
final_data = pd.concat([new_cases, missedData], ignore_index=True)
final_data['date'] = pd.to_datetime(final_data['date'])
final_data.sort_values(by='date', inplace=True)

# Hierarchical clustering results
dataHierarchical = pd.merge(final_data, clustering, how='left', on='Municipalities')
dataHierarchical.fillna(0, inplace=True)
dataHierarchical['date'] = pd.to_datetime(dataHierarchical['date'])
dataHierarchical.Cluster = dataHierarchical.Cluster.astype(str)

pca_variables = pd.read_csv('data\pca_variables.csv')
pca_variables = pd.merge(pca_variables,grp1,on='Municipalities')
variables = pca_variables[[ 'Number', 'demography_0', 'demography_1', 'demography_2',
       'demography_3', 'employment', 'mobility_0', 'mobility_1', 'mobility_2',
       'mobility_3', 'mobility_4', 'mobility_5', 'mobility_6', 'mobility_7',
       'mobility_8', 'ChronicDisease_0', 'ChronicDisease_1',
       'Avg. PI. per person by household position']]

def randomForest(features):
    X = features.iloc[:, 1:]
    y = features.iloc[:, 0]
    rf = RandomForestRegressor()
    rf.fit(X, y)
    names = X.columns
    top_features = rf.feature_importances_[rf.feature_importances_ > 0.05]
    names = names[rf.feature_importances_ > 0.05]
    return top_features, names

top_features, names = randomForest(variables)

# Map plot function

provinces_list = np.append('All', joined_data.Provinces.unique())
municipalities_list =  np.append('All', joined_data.Municipalities)

def plot(data):
    fig = px.choropleth_mapbox(data, geojson=nl_data, locations='id',color='Cluster',
                               mapbox_style="carto-positron",
                               zoom=6, center={"lat": latitude, "lon": longitude},
                               opacity=0.6,
                               labels={'Number': 'Corona Cases'},
                               hover_name='Municipalities',
                               hover_data=['Number'],
                               color_discrete_map = {'1':'#0AA696' ,'2':'#D94B2B','3':'#20638C','4':'#F2B33D','5':'#5E8C30'},
                               )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="white"),
            x=0,
            y=0,
            yanchor="bottom"))
    return fig


def lineplot(data):
    fig = px.line(data, x="date", y='New_cases', line_group='Municipalities',
                  hover_name='Municipalities', color='Cluster', color_discrete_map = {'1':'#0AA696' ,'2':'#D94B2B','3':'#20638C','4':'#F2B33D','5':'#5E8C30'})
    fig.add_annotation(
                x='2020-03-09',
                hovertext="Ban on hand shaking and people in Noord-Brabant<br> to work from home for the next seven days.",
    )
    fig.add_annotation(
                x='2020-03-10',
                hovertext="Travel advice to Italy to code orange,<br> meaning essential travel only."
    )
    fig.add_annotation(
                x='2020-03-12',
                hovertext="The ‘work from home guideline’ is extended to the whole country.<br> Gatherings of more than 100 people are banned"
    )
    fig.add_annotation(
                x='2020-03-13',
                hovertext="Incoming flights are banned from high-risk regions,<br> cited as China, Hong Kong, Iran, South Korea and Italy."
    )
    fig.add_annotation(
                x='2020-03-15',
                hovertext="The ‘intelligent lockdown’ begins. <br>Close schools until April 6. <br>Cafes, restaurants, sports and sex clubs are given less than an hour’s notice to close. <br>People are instructed to keep 1.5 metres apart at all times."
    )
    fig.add_annotation(
                x='2020-03-16',
                hovertext="Funerals are limited to a maximum of 30 mourners."
    )
    fig.add_annotation(
                x='2020-03-17',
                hovertext="Rail operator NS says a scaled-down timetable will run from Saturday,<br>with night trains and most inter-city services cancelled."
    )
    fig.add_annotation(
                x='2020-03-21',
                hovertext="Hundreds of corona patients are transferred from Noord-Brabant to other<br> hospitals to relieve pressure on the province’s healthcare facilities.",
    )
    fig.add_annotation(
                x='2020-03-22',
                hovertext="Coastal resorts close access roads after<br> large numbers of day trippers head out to enjoy the spring sunshine."
    )
    fig.add_annotation(
                x='2020-03-23',
                hovertext="Groups of more than three people in public are banned"
    )
    fig.add_annotation(
                x='2020-03-31',
                hovertext="Lockdown measures are extended until April 28"
    )
    fig.add_annotation(
                x='2020-04-02',
                hovertext="Rutte urges Belgians and Germans to stay away from the Netherlands over the Easter weekend."
    )
    fig.add_annotation(
                x='2020-04-21',
                hovertext="However, the ban on public events is extended to September 1."
    )
    fig.add_annotation(
                x='2020-04-28',
                hovertext="the Rotterdam-Rijnmond region and Limburgse Heuvelland reopen to day trippers."
    )
    fig.add_annotation(
                x='2020-05-1',
                hovertext="Public transport operators call for face masks <br>to be made compulsory in buses, trains and trams."
    )
    fig.add_annotation(
                x='2020-05-8',
                hovertext="The European Union extends the ban on travellers from outside the EU to June 15."
    )
    fig.add_annotation(
                x='2020-05-11',
                hovertext="Some hairdressers reopen at the stroke of midnight as they start <br>to clear the backlog of appointments. Primary schools ban parents from the playground<br> and introduce one-way systems to maintain the 1.5 metre rule.<br> Most pupils are attending school two days a week."
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin={"r": 20, "t": 20, "l": 0, "b": 0}, showlegend=False )
    fig.update_annotations(dict(
            xref="x",
            yref="paper",
            y=0,
            showarrow=True,
            arrowhead=7,
            arrowcolor='tomato',
            arrowsize=2,
            ax=0,
            ay=-300))
    return fig

# Bar plot
def barplot():
    fig = px.bar(x=top_features, y=names, orientation='h')
    fig.update_layout(yaxis={'categoryorder': 'total descending'}, autosize=False,
                      xaxis_title="Score", yaxis_title="Features", showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
    return fig

# app initialize
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
app.config["suppress_callback_exceptions"] = True


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.H6("COVID-19"),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)



app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children="Select Clusters and you will see the the municipalities in different clusters, and ...",
                                ),
                                build_graph_title("Select cluster"),
                                dcc.Dropdown(id='cluster',
                                             options=[{'label': 'All', 'value': 'All'},
                                                      {'label': '1', 'value': '1'},
                                                      {'label': '2', 'value': '2'},
                                                      {'label': '3', 'value': '3'},
                                                      {'label': '4', 'value': '4'},
                                                      {'label': '5', 'value': '5'}], value='All',
                                             ), html.Br(),
                                build_graph_title('Province'),
                                dcc.Dropdown(id='provinces',
                                             options=[{'label': i, 'value': i} for i in provinces_list],
                                             value='All'
                                             ), html.Br(),
                                build_graph_title('Municipality'),
                                dcc.Dropdown(id='municipalities',
                                                     options= [],
                                                     value='All'
                                                     ),html.Br(),
                                build_graph_title('Cases'), html.Br(),
                                dcc.RangeSlider(id='slider',
                                    min=0,
                                    max=500,
                                    step=1,
                                    marks={
                                        0: '0',
                                        100: '100',
                                        200: '200',
                                        300: '300',
                                        400: '400',
                                        500: '>500'
                                    },
                                    value=[0, 500],
                                    tooltip = {'placement': 'top'}
                                                ),

                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # Well map
                        html.Div(
                            id="well-map-container",
                            children=[
                            html.Div(
                                children= [
                                build_graph_title("The Netherlands")
                                ],
                            ),
                            html.Div(
                                children = [dcc.Loading(
                                    # id = 'loading',
                                    children = [dcc.Graph(
                                        id="map",
                                        figure={
                                            "layout": {
                                                "paper_bgcolor": "#192444",
                                                "plot_bgcolor": "#192444",
                                            }
                                        },
                                        config={"scrollZoom": True, "displayModeBar": True},
                                    )],
                                ),
                            ],style={'margin':'5rem 0 5rem'})
                            ],style={'width': '49%','display': 'inline-block','vertical-align': 'middle'}
                        ),
                        # Ternary map
                        html.Div(
                            id="ternary-map-container",
                            children=[
                                html.Div(
                                    id="ternary-header",
                                    children=[
                                        build_graph_title(
                                            "Trend lines"
                                        ),
                                    ],
                                ),
                                html.Div(
                                    children = [dcc.Loading(
                                        # id='loading',
                                        children = [dcc.Graph(
                                            id="trend",
                                            figure={
                                                "layout": {
                                                    "paper_bgcolor": "#192444",
                                                    "plot_bgcolor": "#192444",
                                                }
                                            },
                                            # config={
                                            #     "scrollZoom": True,
                                            #     "displayModeBar": False,
                                            # },
                                        )]
                                    ),
                                ],style={'margin':'5rem 0 0 5rem'})
                            ],style={'width': '49%', 'display': 'inline-block','vertical-align': 'middle'}
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[
                # Formation bar plots
                html.Div(
                    id="form-bar-container",
                    className="six columns",
                    children=[
                        build_graph_title("Important Features"),
                        dcc.Loading(
                            id= 'loading',
                            children = dcc.Graph(id="bar-chart"),
                        ),
                    ],style = {'fontColor':'#1f2c56'}
                ),
            ],
        ),
    ]
)



# Update map
@app.callback(
    Output('map', 'figure'),
    [Input('cluster', 'value'),
    Input('slider', 'value'),
    Input('provinces', 'value'),
    Input('municipalities', 'value')])

def map(cluster, values, province, municipalities):
    df = joined_data
    if cluster != 'All':
        df = joined_data[joined_data.Cluster == str(cluster)]
    if province != 'All':
        df = df[df.Provinces == province]
    if municipalities != 'All':
        df= df[df.Municipalities == municipalities]
    if max(values) == 500:
        df = df[(df['Number'] >= min(values))]
    else:
        df = df[(df['Number'] >= min(values)) & (df['Number'] <= max(values))]
    fig = plot(df)
    return fig

@app.callback(
    Output('municipalities', 'options'),
    [Input('provinces', 'value')])
def set_municipalities(province):
    if province == 'All':
        options = [{'label': i, 'value': i} for i in municipalities_list]
    else:
#        options = [{'label': i, 'value': i} for i in joined_data.loc[joined_data.Provinces == province].Municipalities]
        options_list = np.append('All', joined_data.loc[joined_data.Provinces == province].Municipalities)
        options = [{'label': i, 'value': i} for i in options_list]
    return options

@app.callback(Output('municipalities', 'value'), [Input('provinces', 'value')])
def callback(value):
    return "All"

@app.callback(
    Output('trend', 'figure'),
    [Input('cluster', 'value')])

def trend(cluster):
    if cluster == 'All':
        df = dataHierarchical
        fig = lineplot(df)
        return fig
    else:
        df = dataHierarchical[dataHierarchical.Cluster == str(cluster)]
        fig = lineplot(df)
        return fig

# Range slider text
@app.callback(
    dash.dependencies.Output('output-slider', 'children'),
    [dash.dependencies.Input('slider', 'value')])
def update_output(value):
    return 'Selected range: "{}"'.format(value)

# Update bar plot
@app.callback(
    Output("bar-chart", "figure"),
    [
        Input('cluster', 'value'),
    ],
)
def update_bar(values):
    fig = barplot()
    return fig



# Running the server
if __name__ == "__main__":
    app.run_server(debug=False)
