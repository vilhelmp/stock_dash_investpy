# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import datetime as dt
import pandas as pd
import numpy as np
import investpy


# get a list of all available countries
countries_list = investpy.get_stock_countries()
countries_list.sort()


def get_tickers(country):
    # function to get a dictionary of all tickers/stocks available
    # in a specific country
    try:
        tickers_df = investpy.get_stocks(country=country.lower())
    except:
        tickers_df = pd.DataFrame(data=dict(name=['No results'], symbol=['No results']) )
    # extract names and symbols
    tickers_df.loc[:, 'name'] = tickers_df.loc[:, 'name'].apply(lambda x: x.strip('u200b') )
    
    labels= [', '.join(i) for i in tickers_df.loc[:, ['name','symbol']].values]
    values = tickers_df.loc[:, ['symbol']].values.T[0]
    tickers_dict = [{'label':i, 'value':j} for i,j in zip(labels, values)]
    return tickers_dict


def get_df_ticker(ticker, country, age='2Y'):
    # function to get ticker OLHC prices from name and country
    # by default 2 years back from today
    if 'Y' in age:
        age = float(age.strip('Y'))
        date_since = dt.date.today() - dt.timedelta(days=age*365.25)
    elif 'M' in age:
        date_since = dt.date.today() - dt.timedelta(days=age*31)
    date_since = date_since.strftime('%d/%m/%Y')
    date_to = dt.date.today().strftime('%d/%m/%Y')
    try:
        df = investpy.get_stock_historical_data(stock=ticker, 
                                            country=country, 
                                            from_date=date_since,
                                            to_date=date_to, 
                                            as_json=False, 
                                            order='ascending')
        df.reset_index(inplace=True)
        df.columns = [i.lower() for i in df.columns]
    except:
        # couldn't retrieve data!!
        raise Exception(f"Couldn\'t get data for {ticker} in {country}")
        df = pd.DataFrame([])
    
    return df


#--------------------------------------------------------------------------------

def get_cryptos():
    # function to get dictionary of crypto available at Investing 
    try:
        tickers_df = investpy.get_cryptos()
    except:
        tickers_df = pd.DataFrame(data=dict(name=['No results'], symbol=['No results']) )
    # extract names and symbols
    tickers_df.loc[:, 'name'] = tickers_df.loc[:, 'name'].apply(lambda x: x.strip('u200b') )
    
    labels= [', '.join(i) for i in tickers_df.loc[:, ['name','symbol']].values]
    values = tickers_df.loc[:, ['symbol']].values.T[0]
    tickers_dict = [{'label':i, 'value':j.split(',')[0]} for i,j in zip(labels, labels)]
    return tickers_dict

def get_df_crypto(crypto, age='2Y'):
    # function to get crypto OLHC prices from name
    # by default 2 years back in time
    if 'Y' in age:
        age = float(age.strip('Y'))
        date_since = dt.date.today() - dt.timedelta(days=age*365.25)
    elif 'M' in age:
        date_since = dt.date.today() - dt.timedelta(days=age*31)
    date_since = date_since.strftime('%d/%m/%Y')
    date_to = dt.date.today().strftime('%d/%m/%Y')
    try:
        df = investpy.get_crypto_historical_data(crypto=crypto, 
                                            from_date=date_since,
                                            to_date=date_to, 
                                            as_json=False, 
                                            order='ascending')
        df.reset_index(inplace=True)
        df.columns = [i.lower() for i in df.columns]
    except:
        # couldn't retrieve data!!
        raise Exception(f"Couldn\'t get data for {crypto}")
        df = pd.DataFrame([])
    
    return df



#--------------------------------------------------------------------------------
# Define layout

INCREASING_COLOR = '#17BECF'
DECREASING_COLOR = '#7F7F7F'


layout=dict()

fig = dict(layout=layout )


fig['layout'] = dict()
fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )

# bottom to top
fig['layout']['yaxis'] = dict( domain = [0.01, 0.19], showticklabels = False ) 
fig['layout']['yaxis2'] = dict( domain = [0, 0.2], showticklabels = False)
fig['layout']['yaxis3'] = dict( domain = [0, 0.2], showticklabels = False, anchor="x", overlaying="y2", side="right")
fig['layout']['yaxis4'] = dict( domain = [0.2, 0.8] )

fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )


rangeselector=dict(
    visible = True,
    x = 0, y = 0.9,
    bgcolor = 'rgba(150, 200, 250, 0.4)',
    font = dict( size = 13 ),
    buttons=list([
        dict(count=1,
             label='reset',
             step='all'),
        dict(count=1,
             label='1yr',
             step='year',
             stepmode='backward'),
        dict(count=3,
            label='3 mo',
            step='month',
            stepmode='backward'),
        dict(count=1,
            label='1 mo',
            step='month',
            stepmode='backward'),
        dict(step='all')
    ]),)
    
fig['layout']['xaxis']['rangeselector'] = rangeselector


#--------------------------------------------------------------------------------
# Some simple technical analysis functions

# MA
def movingaverage(interval, window_size=10):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# MACD/EMA
def moving_average_convergence(price, nslow=26, nfast=12):
    emaslow = price.ewm(span=nslow, min_periods=1).mean()
    emafast = price.ewm(span=nfast, min_periods=1).mean()
    result = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    return result

# BB (bollinger bands)
def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band


#--------------------------------------------------------------------------------

# start constructing the WebApp

# Import some nice stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
               )

server = app.server

# When you have tabs, you need this setting to not get a constant 
# exception
app.config['suppress_callback_exceptions'] = True

# DEFINE LAYOUT 

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-stocks', children=[
        dcc.Tab(label='Stocks', value='tab-stocks'),
        dcc.Tab(label='Cryptos', value='tab-cryptos'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    ##############################################################################
    # Tab 1 : Stocks
    if tab == 'tab-stocks':
        return html.Div([
        html.Label('Choose which country for stocks.'),
        dcc.Dropdown(
            options=[{'label': k.capitalize(), 'value': k} for k in countries_list],
            id='country-dropdown',
            #value='Sweden',
            multi=False
        ),

        html.Label('Choose stocks to plot'),
        dcc.Dropdown(
            options=get_tickers('sweden'),
            id='tickers-dropdown',
            #value=[''],
            multi=True
        ),

        html.Hr(),

        html.Div(id='display-selected-values'),

        html.Label('Plot of selected stock(s)'),
        dcc.Graph(id='candlestick-graphic', style={"height": "600px"}),
    ])

    ##############################################################################
    # Tab 2 : Cryptocurrencies
    elif tab == 'tab-cryptos':
        return html.Div([
        html.Label('Choose crypto to plot'),
        dcc.Dropdown(
            options=get_cryptos(),
            id='cryptos-dropdown',
            #value=[''],
            multi=True
        ),

        html.Hr(),

        html.Label('Plot of selected crypto(s)'),
        dcc.Graph(id='candlestick-crypto-graphic'),
        ])
    
# DEFINE CALLBACKS for TAB STOCKS
##############################################################################
@app.callback(
    [Output('tickers-dropdown', 'options'),
     Output('tickers-dropdown', 'value')],
    [Input('country-dropdown', 'value')])
def set_ticks_options(selected_country):
    return [get_tickers(selected_country),'']

@app.callback(
    Output('display-selected-values', 'children'),
    [Input('country-dropdown', 'value'),
     Input('tickers-dropdown', 'value')])
def set_display_children(selected_country, selected_tick):    
    if selected_tick:
        # split the selected_tick label and get the symbol
        selected_tick = selected_tick[0]
        return f"{selected_tick} is a stock in {selected_country}",
    else:
        return 'No stocks selected.'

@app.callback(
    Output('candlestick-graphic', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('tickers-dropdown', 'value'),
    ])
def update_graph(selected_country, selected_tick):
    #dff = df[df['Year'] == year_value]
    if not selected_tick:
        return {}
    #ticker = ticker_name[0]
    # split the selected_tick label and get the symbol
    selected_tick = selected_tick[0]
    df = get_df_ticker(selected_tick, selected_country)
    
    #define data
    data = [ dict(
        type = 'candlestick',
        open = df.open,
        high = df.high,
        low = df.low,
        close = df.close,
        x = df.date,
        yaxis = 'y4',
        name = str(selected_tick),
        increasing = dict( line = dict( color = INCREASING_COLOR ) ),
        decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
    ) ]
    # overwrite data, whatever it is
    
    fig['data'] = data
    
    # Calculate and plot moving averages
    # calculate MA
    mv_y = movingaverage(df.close)
    mv_x = list(df.date)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

    fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#E377C2' ),
                             yaxis = 'y4', name='Moving Average' ) )

    # Define OLHC candlestick colors
    colors = []

    for i in range(len(df.close)):
        if i != 0:
            if df.close[i] > df.close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)


    # Plot the volume
    fig['data'].append( dict( x=df.date, y=df.volume,                         
                             marker=dict( color=colors ),
                             type='bar', yaxis='y', name='Volume' ) )

    # plot MACD graph
    
    macd_y = moving_average_convergence(df.close)
    macd_x = list(df.date)

    # Clip the ends
    macd_x = macd_x[5:-5]
    macd_y = macd_y[5:-5]

    fig['data'].append( dict( x=macd_x, y=macd_y.MACD,                         
                             marker=dict( color=colors ),
                             type='bar', yaxis='y2', name='MACD' ) )

    
    fig['data'].append( dict( x=macd_x, y=macd_y.emaSlw, type='scatter', yaxis='y3', 
                             line = dict( width = 2 ),
                             marker=dict(color='indianred'), hoverinfo='none', 
                             legendgroup='MACD', name='MACD slow') )

    fig['data'].append( dict( x=macd_x, y=macd_y.emaFst, type='scatter', yaxis='y3', 
                             line = dict( width = 2 ),
                             marker=dict(color='cornflowerblue'), hoverinfo='none', 
                             legendgroup='MACD', name='MACD fast') )
    
    # Calculate and plot the bollinger bands
    bb_avg, bb_upper, bb_lower = bbands(df.close)

    fig['data'].append( dict( x=df.date, y=bb_upper, type='scatter', yaxis='y4', 
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none', 
                             legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.date, y=bb_lower, type='scatter', yaxis='y4',
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none',
                             legendgroup='Bollinger Bands', showlegend=False ) )
    
    # Attempt att getting y-axis to rescale automatically with range slider
    #fig['layout']['yaxis'] = {
    #    #"domain": [0, 1], 
    #    #"title": "Price",
    #    'yaxis':'y2',
    #    "autorange": True,
    #    "fixedrange":False
    #  }
    return go.Figure(fig)


# DEFINE CALLBACKS TAB CRYPTOS
##############################################################################


@app.callback(
    Output('candlestick-crypto-graphic', 'figure'),
    [Input('cryptos-dropdown', 'value'),
    ])
def update_graph_crypto(selected_crypto):
    #dff = df[df['Year'] == year_value]
    if not selected_crypto:
        return {}
    #ticker = ticker_name[0]
    # split the selected_tick label and get the symbol
    selected_crypto = selected_crypto[0]
    df = get_df_crypto(selected_crypto)
    
    #define data
    data = [ dict(
        type = 'candlestick',
        open = df.open,
        high = df.high,
        low = df.low,
        close = df.close,
        x = df.date,
        yaxis = 'y4',
        name = str(selected_crypto),
        increasing = dict( line = dict( color = INCREASING_COLOR ) ),
        decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
    ) ]
    # overwrite data, whatever it is
    
    fig['data'] = data
    
    # calculate MA
    mv_y = movingaverage(df.close)
    mv_x = list(df.date)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

    fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#E377C2' ),
                             yaxis = 'y4', name='Moving Average' ) )


    colors = []

    for i in range(len(df.close)):
        if i != 0:
            if df.close[i] > df.close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)



    fig['data'].append( dict( x=df.date, y=df.volume,                         
                             marker=dict( color=colors ),
                             type='bar', yaxis='y2', name='Volume' ) )


    bb_avg, bb_upper, bb_lower = bbands(df.close)

    fig['data'].append( dict( x=df.date, y=bb_upper, type='scatter', yaxis='y4', 
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none', 
                             legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.date, y=bb_lower, type='scatter', yaxis='y4',
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none',
                             legendgroup='Bollinger Bands', showlegend=False ) )


    return go.Figure(fig)

##############################################################################


if __name__ == '__main__':
    app.run_server(debug=False,
                  #dev_tools_ui=False,dev_tools_props_check=False,
                  )



