"""
Trading Strategy Dashboard

A Dash-based UI for visualizing backtesting strategies and market analysis.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

# Import strategy modules
from strategies.tqqq_ma200_strategy import TQQQMA200Strategy, StrategyParams
from strategies.signal_checker import SignalChecker
from strategies.leveraged_etf_comparison import LeveragedETFComparison, ComparisonParams
from strategies.liquidity_analysis import LiquidityAnalysis, LiquidityParams


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "Trading Strategy Dashboard"


# =====================================================
# LAYOUT COMPONENTS
# =====================================================

def create_navbar():
    """Create navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Trading Strategy Dashboard", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Refresh Data", id="refresh-btn", href="#", className="btn btn-outline-light")),
            ], className="ms-auto"),
        ]),
        color="dark",
        dark=True,
        className="mb-4"
    )


def create_signal_checker_tab():
    """Create signal checker tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Daily Signal Checker", className="mb-3"),
                html.P("Check today's market conditions for BUY/SELL signals."),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Today's Signal"),
                    dbc.CardBody(id="signal-status-card")
                ], className="mb-3")
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Market Data"),
                    dbc.CardBody(id="market-data-card")
                ], className="mb-3")
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Condition Check"),
                    dbc.CardBody(id="condition-check-card")
                ], className="mb-3")
            ], md=4),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="signal-chart")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H5("Recent Price History", className="mt-4"),
                html.Div(id="recent-history-table")
            ])
        ])
    ])


def create_backtest_tab():
    """Create TQQQ MA200 backtest tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("TQQQ MA200 Strategy Backtest", className="mb-3"),
                html.P("Backtest the MA200 strategy with customizable parameters."),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Parameters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Start Date"),
                                dbc.Input(id="backtest-start-date", type="text", value="2015-01-01"),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Initial Capital ($)"),
                                dbc.Input(id="backtest-capital", type="number", value=100000),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Buy Threshold"),
                                dbc.Input(id="backtest-buy-threshold", type="number", value=1.04, step=0.01),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Sell Threshold"),
                                dbc.Input(id="backtest-sell-threshold", type="number", value=0.97, step=0.01),
                            ], md=3),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Run Backtest", id="run-backtest-btn", color="primary", className="mt-3"),
                            ])
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics"),
                    dbc.CardBody(id="backtest-metrics-card")
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Trade Log"),
                    dbc.CardBody(id="trade-log-card", style={"maxHeight": "300px", "overflow": "auto"})
                ])
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="backtest-chart")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="backtest-drawdown-chart")
            ])
        ])
    ])


def create_etf_comparison_tab():
    """Create ETF comparison tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Leveraged ETF Comparison", className="mb-3"),
                html.P("Compare different ETFs using QQQ-based entry/exit signals."),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Comparison Settings"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Analysis Period"),
                                dcc.Dropdown(
                                    id="etf-period-dropdown",
                                    options=[
                                        {"label": "Long-term (2018-present)", "value": "long"},
                                        {"label": "Recent (2023-present)", "value": "recent"},
                                    ],
                                    value="long"
                                ),
                            ], md=4),
                            dbc.Col([
                                dbc.Button("Run Comparison", id="run-etf-comparison-btn", color="primary", className="mt-4"),
                            ], md=2),
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H5("Strategy Performance Comparison"),
                html.Div(id="etf-comparison-table")
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="etf-performance-chart")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="etf-buyhold-chart")
            ])
        ])
    ])


def create_liquidity_tab():
    """Create liquidity analysis tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("US Market Liquidity Analysis", className="mb-3"),
                html.P("Analyze daily market liquidity with confidence intervals."),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Current Liquidity Status"),
                    dbc.CardBody(id="liquidity-status-card")
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Volume Metrics"),
                    dbc.CardBody(id="liquidity-volume-card")
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Volatility Metrics"),
                    dbc.CardBody(id="liquidity-volatility-card")
                ])
            ], md=4),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Current Values vs 95% Confidence Intervals"),
                    dbc.CardBody(id="liquidity-ci-table")
                ])
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="liquidity-dashboard-chart")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="liquidity-ci-chart")
            ])
        ])
    ])


# Main layout
app.layout = html.Div([
    create_navbar(),
    dbc.Container([
        dcc.Store(id="data-store"),
        dcc.Tabs(id="main-tabs", value="signal-tab", children=[
            dcc.Tab(label="Daily Signal", value="signal-tab"),
            dcc.Tab(label="TQQQ Backtest", value="backtest-tab"),
            dcc.Tab(label="ETF Comparison", value="etf-tab"),
            dcc.Tab(label="Liquidity Analysis", value="liquidity-tab"),
        ], className="mb-4"),
        html.Div(id="tab-content")
    ], fluid=True)
])


# =====================================================
# CALLBACKS
# =====================================================

@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value")
)
def render_tab_content(tab):
    """Render content based on selected tab."""
    if tab == "signal-tab":
        return create_signal_checker_tab()
    elif tab == "backtest-tab":
        return create_backtest_tab()
    elif tab == "etf-tab":
        return create_etf_comparison_tab()
    elif tab == "liquidity-tab":
        return create_liquidity_tab()
    return html.Div("Select a tab")


# Signal Checker Callbacks
@callback(
    [Output("signal-status-card", "children"),
     Output("market-data-card", "children"),
     Output("condition-check-card", "children"),
     Output("signal-chart", "figure"),
     Output("recent-history-table", "children")],
    Input("main-tabs", "value"),
    prevent_initial_call=False
)
def update_signal_checker(tab):
    """Update signal checker tab."""
    if tab != "signal-tab":
        return [html.Div(), html.Div(), html.Div(), go.Figure(), html.Div()]

    checker = SignalChecker()
    summary = checker.get_summary_dict()

    # Signal status card
    signal_color = {"BUY": "success", "SELL": "danger", "HOLD": "secondary"}
    signal_status = html.Div([
        html.H2(summary['signal'], className=f"text-{signal_color.get(summary['signal'], 'secondary')}"),
        html.P(f"Current Position: {summary['position']}"),
        html.P(f"Last Action: {summary['last_action'] or 'None'} on {summary['last_action_date'] or 'N/A'}"),
    ])

    # Market data card
    market_data = html.Div([
        html.P(f"Date: {summary['date']}"),
        html.P(f"QQQ Close: ${summary['qqq_close']:.2f}"),
        html.P(f"QQQ Daily Change: {summary['qqq_daily_return']:+.2f}%"),
        html.P(f"TQQQ Close: ${summary['tqqq_close']:.2f}"),
        html.Hr(),
        html.P(f"MA200: ${summary['ma200']:.2f}"),
        html.P(f"Buy Level: ${summary['buy_level']:.2f}"),
        html.P(f"Sell Level: ${summary['sell_level']:.2f}"),
    ])

    # Condition check card
    cond = summary['conditions']
    condition_check = html.Div([
        html.H6("BUY Conditions:"),
        html.P([
            html.Span("✓ " if cond['above_buy_level'] else "✗ ", className=f"text-{'success' if cond['above_buy_level'] else 'danger'}"),
            f"QQQ > Buy Level"
        ]),
        html.P([
            html.Span("✓ " if cond['daily_loss_met'] else "✗ ", className=f"text-{'success' if cond['daily_loss_met'] else 'danger'}"),
            f"Daily loss >= 1%"
        ]),
        html.Hr(),
        html.H6("SELL Condition:"),
        html.P([
            html.Span("✓ " if cond['below_sell_level'] else "✗ ", className=f"text-{'success' if cond['below_sell_level'] else 'danger'}"),
            f"QQQ < Sell Level"
        ]),
    ])

    # Chart
    fig = checker.create_chart(days=60)

    # Recent history table
    recent = checker.get_recent_history(10)
    table = dbc.Table.from_dataframe(
        recent.reset_index().round(2),
        striped=True, bordered=True, hover=True, size="sm"
    )

    return [signal_status, market_data, condition_check, fig, table]


# Backtest Callbacks
@callback(
    [Output("backtest-metrics-card", "children"),
     Output("trade-log-card", "children"),
     Output("backtest-chart", "figure"),
     Output("backtest-drawdown-chart", "figure")],
    Input("run-backtest-btn", "n_clicks"),
    [State("backtest-start-date", "value"),
     State("backtest-capital", "value"),
     State("backtest-buy-threshold", "value"),
     State("backtest-sell-threshold", "value")],
    prevent_initial_call=True
)
def run_backtest(n_clicks, start_date, capital, buy_thresh, sell_thresh):
    """Run backtest and update results."""
    if not n_clicks:
        return [html.Div(), html.Div(), go.Figure(), go.Figure()]

    params = StrategyParams(
        start_date=start_date,
        initial_capital=capital,
        buy_threshold=buy_thresh,
        sell_threshold=sell_thresh
    )

    strategy = TQQQMA200Strategy(params)
    results = strategy.run_full_analysis()
    metrics = results['metrics']

    # Metrics card
    metrics_content = html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Strategy"),
                html.P(f"Total Return: {metrics['strategy']['total_return']:.2f}%"),
                html.P(f"Ann. Return: {metrics['strategy']['ann_return']:.2f}%"),
                html.P(f"Sharpe: {metrics['strategy']['sharpe']:.2f}"),
                html.P(f"Max DD: {metrics['strategy']['max_drawdown']:.2f}%"),
                html.P(f"Final Value: ${metrics['strategy']['final_value']:,.0f}"),
            ]),
            dbc.Col([
                html.H6("Buy & Hold TQQQ"),
                html.P(f"Total Return: {metrics['buy_hold']['total_return']:.2f}%"),
                html.P(f"Ann. Return: {metrics['buy_hold']['ann_return']:.2f}%"),
                html.P(f"Sharpe: {metrics['buy_hold']['sharpe']:.2f}"),
                html.P(f"Max DD: {metrics['buy_hold']['max_drawdown']:.2f}%"),
                html.P(f"Final Value: ${metrics['buy_hold']['final_value']:,.0f}"),
            ])
        ]),
        html.Hr(),
        html.P(f"Time in Market: {metrics['time_in_market']:.2f}%"),
        html.P(f"Number of Trades: {metrics['num_trades']}"),
    ])

    # Trade log
    trade_log = results['trade_log'].copy()
    trade_log['QQQ_Daily_Return'] = (trade_log['QQQ_Daily_Return'] * 100).round(2).astype(str) + '%'
    trade_log = trade_log.round(2)
    trade_table = dbc.Table.from_dataframe(
        trade_log.reset_index(),
        striped=True, bordered=True, hover=True, size="sm"
    )

    # Charts
    portfolio_fig = strategy.create_portfolio_chart()
    drawdown_fig = strategy.create_drawdown_chart()

    return [metrics_content, trade_table, portfolio_fig, drawdown_fig]


# ETF Comparison Callbacks
@callback(
    [Output("etf-comparison-table", "children"),
     Output("etf-performance-chart", "figure"),
     Output("etf-buyhold-chart", "figure")],
    Input("run-etf-comparison-btn", "n_clicks"),
    State("etf-period-dropdown", "value"),
    prevent_initial_call=True
)
def run_etf_comparison(n_clicks, period):
    """Run ETF comparison and update results."""
    if not n_clicks:
        return [html.Div(), go.Figure(), go.Figure()]

    comparison = LeveragedETFComparison()

    if period == "long":
        comparison.run_long_term_analysis()
        assets = comparison.DEFAULT_ASSETS_LONG
    else:
        comparison.run_recent_analysis()
        assets = comparison.DEFAULT_ASSETS_RECENT

    # Comparison table
    table_df = comparison.get_comparison_table(assets)
    table = dbc.Table.from_dataframe(
        table_df.round(2),
        striped=True, bordered=True, hover=True
    )

    # Charts
    perf_fig = comparison.create_performance_chart(assets)
    bh_fig = comparison.create_buy_hold_chart(assets)

    return [table, perf_fig, bh_fig]


# Liquidity Analysis Callbacks
@callback(
    [Output("liquidity-status-card", "children"),
     Output("liquidity-volume-card", "children"),
     Output("liquidity-volatility-card", "children"),
     Output("liquidity-ci-table", "children"),
     Output("liquidity-dashboard-chart", "figure"),
     Output("liquidity-ci-chart", "figure")],
    Input("main-tabs", "value"),
    prevent_initial_call=False
)
def update_liquidity_analysis(tab):
    """Update liquidity analysis tab."""
    if tab != "liquidity-tab":
        return [html.Div(), html.Div(), html.Div(), html.Div(), go.Figure(), go.Figure()]

    analysis = LiquidityAnalysis()
    results = analysis.run_full_analysis()
    status = results['current_status']

    # Status card
    color_map = {"GREEN": "success", "YELLOW": "warning", "ORANGE": "warning", "RED": "danger"}
    status_card = html.Div([
        html.H4(f"{status['liquidity_index']:.1f}/100", className=f"text-{color_map.get(status['color'], 'secondary')}"),
        html.P(f"Regime: {status['regime']}"),
        html.P(status['assessment'], className="small"),
        html.Hr(),
        html.P(f"Date: {status['date']}"),
    ])

    # Volume card
    volume_card = html.Div([
        html.P(f"SPY Volume: {status['spy_volume']:.1f}M"),
        html.P(f"Volume Ratio: {status['volume_ratio']:.2f}x"),
        html.P(f"Dollar Volume: ${status['dollar_volume']:.2f}B"),
    ])

    # Volatility card
    volatility_card = html.Div([
        html.P(f"VIX: {status['vix']:.2f}"),
        html.P(f"VIX Percentile: {status['vix_percentile']:.1f}%" if status['vix_percentile'] else "VIX Percentile: N/A"),
        html.P(f"Realized Vol: {status['realized_vol']:.1f}%"),
        html.P(f"VIX-RV Spread: {status['vix_rv_spread']:.1f}"),
    ])

    # CI table
    ci_rows = []
    for check in status['status_vs_ci']:
        status_color = {"NORMAL": "dark", "ABOVE CI": "danger", "BELOW CI": "primary"}
        ci_rows.append(html.Tr([
            html.Td(check['metric']),
            html.Td(f"{check['current']:.2f}"),
            html.Td(f"{check['mean']:.2f}"),
            html.Td(f"[{check['ci_low']:.2f}, {check['ci_high']:.2f}]"),
            html.Td(check['status'], className=f"text-{status_color.get(check['status'], 'dark')}"),
        ]))

    ci_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Metric"), html.Th("Current"), html.Th("Mean"), html.Th("95% CI"), html.Th("Status")
        ])),
        html.Tbody(ci_rows)
    ], striped=True, bordered=True, hover=True, size="sm")

    # Charts
    dashboard_fig = analysis.create_dashboard_chart()
    ci_fig = analysis.create_ci_chart()

    return [status_card, volume_card, volatility_card, ci_table, dashboard_fig, ci_fig]


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    print("Starting Trading Strategy Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, host="127.0.0.1", port=8050)
