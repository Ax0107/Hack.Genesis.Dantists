import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from heatmap import calmap
import matplotlib.pyplot as plt

promo_history = pd.read_excel('promo_history.xlsx', engine='openpyxl')
sales_history = pd.read_csv('sales_history.csv')

promo_history.drop(columns=['Unnamed: 0', 0], inplace=True)
sales_history.drop(columns=['Unnamed: 0'], inplace=True)

sales_history['sale_dt'] = sales_history['sale_dt'].apply(lambda x: datetime.datetime.fromisoformat(x))

products = np.unique(sales_history['skutertiaryid'].values)

profits = []
for i in range(len(promo_history)):
    data = promo_history.iloc[i]
    start = data['start_dttm']
    end = data['end_dttm']
    product = data['skutertiaryid']

    history = sales_history[(sales_history['sale_dt'] >= start) & (sales_history['sale_dt'] <= end) & (
            sales_history['skutertiaryid'] == product)]
    earn = history['salerevenuerub'].sum()
    expenses = (history['salerevenuerub'] * data['chaindiscountvalue']).sum()
    profit = (earn - expenses) / (end - start).days
    profits.append(profit)


def get_monthly_profit(year, data):
    df = data[(data['sale_dt'] < datetime.datetime.fromisoformat(f'{str(int(year) + 1)}-01-01')) \
              & (data['sale_dt'] > datetime.datetime.fromisoformat(f'{str(int(year) - 1)}-12-30'))]
    monthly_profit = []
    for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        temp_data = df[(df['sale_dt'] <= datetime.datetime.fromisoformat(f'{year}-{month}-{28}')) \
                       & (df['sale_dt'] >= datetime.datetime.fromisoformat(f'{year}-{month}-01'))]
        monthly_profit.append(temp_data['salerevenuerub'].sum() / 30)

    return np.array(monthly_profit)


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
    dbc.themes.BOOTSTRAP,
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(
        children=[
            html.Div(
                children=[
                    html.P(children="👨‍💻", className="header-emoji"),
                    html.H1(
                        children="Dantists Analytics", className="header-title"
                    ),
                    html.P(
                        children="Решение команды Dantists по задаче максимизации выгоды с проведения промо-акций",
                        className="header-description",
                    ),
                ],
                className="header",
            )]
    ),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(children="Товар", className="menu-title"),

                    dcc.Dropdown(
                        id="region-filter",
                        options=[
                            {"label": f'ID {i}', "value": f'ID {i}'}
                            for i in products
                        ],
                        value=f"ID {products[0]}",
                        clearable=False,
                        className="dropdown",
                    ),
                ]
            ),
            html.Div(
                children=[
                    html.Div(children="Бюджет", className="menu-title"),
                    dbc.Input(id="input-budget", className='button', value=100000,
                              placeholder="Введите бюджет промо-акций"),
                ],
            ),
            html.Div(
                children=[
                    html.Div(
                        children="Date Range",
                        className="menu-title"
                    ),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=datetime.datetime.fromisoformat('2018-01-01'),
                        max_date_allowed=datetime.datetime.fromisoformat('2021-01-01'),
                        start_date=datetime.datetime.fromisoformat('2020-01-01'),
                        end_date=datetime.datetime.fromisoformat('2020-02-01'),
                    ),
                ]
            ),
        ],
        className="menu",
    ),
    html.Div([
    dbc.Button('Рассчитать', id='btn-calculate', n_clicks=0,
               style={'marginTop': '50px', 'width': '100%'}),
    ], className='menu'),

    html.Div(
        children=[
            html.Div([], className='card', id='calmap'),

            html.Div(
                children=[], id='graphs'
            ),
        ],
        className="wrapper",
    ),

    dcc.Store(id='output-sales-data'),
    html.Br(),
    html.Div(id='output-2'),

])


@app.callback([Output('graphs', 'children'),
              Output('calmap', 'children')],
              [Input('btn-calculate', 'n_clicks')],
              [State('input-budget', 'value')])
def update_output(n, bugdet):
    if not n:
        return [], []

    children = []

    for year in range(2018, 2022):
        profit_figure = {
            "data": [
                {
                    "y": get_monthly_profit(str(year), sales_history),
                    "type": "lines",
                    "hovertemplate": "$%{y:.2f}<extra></extra>",
                },
            ],
            "layout": {
                "title": {
                    "text": f"Прибыль в {year} году",
                    "x": 0.05,
                    "xanchor": "left",
                },
                "xaxis": {"fixedrange": True},
                "yaxis": {"tickprefix": "$", "fixedrange": True},
                "colorway": ["#17B897"],
            },
        }
        graph = dcc.Graph(id=f"profit-chart-{year}", figure=profit_figure),
        children.append(graph[0])

    fig = plt.figure(figsize=(8, 4.5), dpi=100)
    X = np.linspace(-1, 1, 53 * 7)
    ax = plt.subplot(311, xlim=[0, 53], ylim=[0, 7], frameon=False, aspect=1)
    I = 1.2 - np.cos(X.ravel()) + np.random.normal(0, .2, X.size)
    calmap(ax, 2017, I.reshape(53, 7).T)
    plt.savefig("assets/calendar-heatmap.png", dpi=300)

    return children, html.Img(src=app.get_asset_url("calendar-heatmap.png"), width='auto')


if __name__ == '__main__':
    app.run_server(debug=True)
