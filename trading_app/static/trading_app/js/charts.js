/**
 * Chart rendering and management using Plotly.js
 */

// Default chart layout configuration
const defaultLayout = {
    autosize: true,
    margin: { l: 50, r: 50, t: 30, b: 50 },
    hovermode: 'x unified',
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    font: {
        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        size: 12,
        color: '#212529'
    },
    xaxis: {
        gridcolor: '#e9ecef',
        showgrid: true
    },
    yaxis: {
        gridcolor: '#e9ecef',
        showgrid: true
    },
    legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: -0.3,
        xanchor: 'center',
        x: 0.5
    }
};

// Default chart config
const defaultConfig = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false
};

/**
 * Clear container and prepare for new chart
 */
function clearChartContainer(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
    }
}

/**
 * Load all charts for a given ticker
 */
function loadAllCharts(ticker) {
    console.log(`Loading charts for ticker: ${ticker}`);

    // Show loading state for all charts
    const chartIds = [
        'price-forecast-chart',
        'moving-averages-chart',
        'rsi-chart',
        'macd-chart',
        'bollinger-chart',
        'atr-chart',
        'obv-chart',
        'stochastic-chart',
        'ichimoku-chart',
        'adx-chart',
        'vwap-chart'
    ];

    chartIds.forEach(id => showLoading(id));

    // Fetch chart data from API
    fetch(`/api/chart-data/${ticker}/`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch chart data');
            }
            return response.json();
        })
        .then(data => {
            // Render all charts
            renderPriceForecastChart(data.price_forecast);
            renderMovingAveragesChart(data.moving_averages);
            renderRSIChart(data.rsi);
            renderMACDChart(data.macd);
            renderBollingerChart(data.bollinger);
            renderATRChart(data.atr);
            renderOBVChart(data.obv);
            renderStochasticChart(data.stochastic);
            renderIchimokuChart(data.ichimoku);
            renderADXChart(data.adx);
            renderVWAPChart(data.vwap);
        })
        .catch(error => {
            console.error('Error loading charts:', error);
            chartIds.forEach(id => showError(id, 'Failed to load chart data'));
        });
}

/**
 * Render Price Forecast Chart
 */
function renderPriceForecastChart(data) {
    clearChartContainer('price-forecast-chart');

    const currency = data.currency;

    const traces = [
        {
            x: data.historical.dates,
            y: data.historical.close,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical Close',
            line: { color: '#0d6efd', width: 2 }
        },
        {
            x: data.forecast.dates,
            y: data.forecast.close,
            type: 'scatter',
            mode: 'lines',
            name: 'Forecast Close',
            line: { color: '#dc3545', width: 2, dash: 'dot' }
        },
        {
            x: data.ma_10.dates,
            y: data.ma_10.values,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 10',
            line: { color: '#fd7e14', width: 1, dash: 'dot' }
        }
    ];

    // Add confidence interval band
    if (data.confidence_intervals) {
        // Upper bound
        traces.push({
            x: data.confidence_intervals.dates,
            y: data.confidence_intervals.upper,
            type: 'scatter',
            mode: 'lines',
            name: '90% CI Upper',
            line: { color: 'rgba(220, 53, 69, 0.3)', width: 1, dash: 'dash' },
            showlegend: true
        });

        // Lower bound
        traces.push({
            x: data.confidence_intervals.dates,
            y: data.confidence_intervals.lower,
            type: 'scatter',
            mode: 'lines',
            name: '90% CI Lower',
            line: { color: 'rgba(220, 53, 69, 0.3)', width: 1, dash: 'dash' },
            fill: 'tonexty',  // Fill area between this and previous trace
            fillcolor: 'rgba(220, 53, 69, 0.1)',
            showlegend: true
        });
    }

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: `Price (${currency})` },
        hovermode: 'x unified'
    };

    Plotly.newPlot('price-forecast-chart', traces, layout, defaultConfig);
}

/**
 * Render Moving Averages Chart
 */
function renderMovingAveragesChart(data) {
    clearChartContainer('moving-averages-chart');

    const traces = [
        {
            x: data.historical.dates,
            y: data.historical.close,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical Close',
            line: { color: '#0d6efd', width: 2 }
        },
        {
            x: data.ma_200.dates,
            y: data.ma_200.values,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 200',
            line: { color: '#ffc107', width: 2 }
        },
        {
            x: data.ma_50.dates,
            y: data.ma_50.values,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 50',
            line: { color: '#fd7e14', width: 2, dash: 'dot' }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'Price (USD)' }
    };

    Plotly.newPlot('moving-averages-chart', traces, layout, defaultConfig);
}

/**
 * Render RSI Chart
 */
function renderRSIChart(data) {
    clearChartContainer('rsi-chart');

    const traces = [
        {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI',
            line: { color: '#fd7e14', width: 2 }
        },
        // Overbought line (70)
        {
            x: data.dates,
            y: Array(data.dates.length).fill(70),
            type: 'scatter',
            mode: 'lines',
            name: 'Overbought (70)',
            line: { color: '#dc3545', width: 1, dash: 'dash' },
            showlegend: false
        },
        // Oversold line (30)
        {
            x: data.dates,
            y: Array(data.dates.length).fill(30),
            type: 'scatter',
            mode: 'lines',
            name: 'Oversold (30)',
            line: { color: '#198754', width: 1, dash: 'dash' },
            showlegend: false
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'RSI', range: [0, 100] }
    };

    Plotly.newPlot('rsi-chart', traces, layout, defaultConfig);
}

/**
 * Render MACD Chart
 */
function renderMACDChart(data) {
    clearChartContainer('macd-chart');

    const traces = [
        {
            x: data.dates,
            y: data.macd,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: { color: '#fd7e14', width: 2 }
        },
        {
            x: data.dates,
            y: data.macd_signal,
            type: 'scatter',
            mode: 'lines',
            name: 'Signal',
            line: { color: '#ffc107', width: 2 }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'MACD' }
    };

    Plotly.newPlot('macd-chart', traces, layout, defaultConfig);
}

/**
 * Render Bollinger Bands Chart
 */
function renderBollingerChart(data) {
    clearChartContainer('bollinger-chart');

    const traces = [
        {
            x: data.dates,
            y: data.bb_upper,
            type: 'scatter',
            mode: 'lines',
            name: 'BB Upper',
            line: { color: '#fd7e14', width: 2 }
        },
        {
            x: data.dates,
            y: data.bb_lower,
            type: 'scatter',
            mode: 'lines',
            name: 'BB Lower',
            line: { color: '#ffc107', width: 2 }
        },
        {
            x: data.dates,
            y: data.close,
            type: 'scatter',
            mode: 'lines',
            name: 'Close',
            line: { color: '#0d6efd', width: 2 }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'Price (USD)' }
    };

    Plotly.newPlot('bollinger-chart', traces, layout, defaultConfig);
}

/**
 * Render ATR Chart
 */
function renderATRChart(data) {
    clearChartContainer('atr-chart');

    const traces = [
        {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            name: 'ATR',
            line: { color: '#fd7e14', width: 2 }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'ATR' }
    };

    Plotly.newPlot('atr-chart', traces, layout, defaultConfig);
}

/**
 * Render OBV Chart
 */
function renderOBVChart(data) {
    clearChartContainer('obv-chart');

    const traces = [
        {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            name: 'OBV',
            line: { color: '#fd7e14', width: 2 }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'OBV' }
    };

    Plotly.newPlot('obv-chart', traces, layout, defaultConfig);
}

/**
 * Render Stochastic Oscillator Chart
 */
function renderStochasticChart(data) {
    clearChartContainer('stochastic-chart');

    const traces = [
        {
            x: data.dates,
            y: data.stoch_k,
            type: 'scatter',
            mode: 'lines',
            name: 'STOCH-K',
            line: { color: '#fd7e14', width: 2 }
        },
        {
            x: data.dates,
            y: data.stoch_d,
            type: 'scatter',
            mode: 'lines',
            name: 'STOCH-D',
            line: { color: '#ffc107', width: 2 }
        },
        // Overbought line (80)
        {
            x: data.dates,
            y: Array(data.dates.length).fill(80),
            type: 'scatter',
            mode: 'lines',
            name: 'Overbought (80)',
            line: { color: '#dc3545', width: 1, dash: 'dash' },
            showlegend: false
        },
        // Oversold line (20)
        {
            x: data.dates,
            y: Array(data.dates.length).fill(20),
            type: 'scatter',
            mode: 'lines',
            name: 'Oversold (20)',
            line: { color: '#198754', width: 1, dash: 'dash' },
            showlegend: false
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'Stochastic', range: [0, 100] }
    };

    Plotly.newPlot('stochastic-chart', traces, layout, defaultConfig);
}

/**
 * Render Ichimoku Cloud Chart
 */
function renderIchimokuChart(data) {
    clearChartContainer('ichimoku-chart');

    const traces = [
        {
            x: data.dates,
            y: data.ichimoku_senkou_a,
            type: 'scatter',
            mode: 'lines',
            name: 'Cloud Boundary A',
            line: { color: '#fd7e14', width: 2 }
        },
        {
            x: data.dates,
            y: data.ichimoku_senkou_b,
            type: 'scatter',
            mode: 'lines',
            name: 'Cloud Boundary B',
            line: { color: '#ffc107', width: 2 }
        },
        {
            x: data.dates,
            y: data.close,
            type: 'scatter',
            mode: 'lines',
            name: 'Close',
            line: { color: '#0d6efd', width: 2 }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'Price (USD)' }
    };

    Plotly.newPlot('ichimoku-chart', traces, layout, defaultConfig);
}

/**
 * Render ADX Chart
 */
function renderADXChart(data) {
    clearChartContainer('adx-chart');

    const traces = [
        {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            name: 'ADX',
            line: { color: '#fd7e14', width: 2 }
        },
        // Strong trend line (40)
        {
            x: data.dates,
            y: Array(data.dates.length).fill(40),
            type: 'scatter',
            mode: 'lines',
            name: 'Strong Trend (40)',
            line: { color: '#198754', width: 1, dash: 'dash' },
            showlegend: false
        },
        // Weak trend line (20)
        {
            x: data.dates,
            y: Array(data.dates.length).fill(20),
            type: 'scatter',
            mode: 'lines',
            name: 'Weak Trend (20)',
            line: { color: '#dc3545', width: 1, dash: 'dash' },
            showlegend: false
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'ADX', range: [0, 100] }
    };

    Plotly.newPlot('adx-chart', traces, layout, defaultConfig);
}

/**
 * Render VWAP Chart
 */
function renderVWAPChart(data) {
    clearChartContainer('vwap-chart');

    const traces = [
        {
            x: data.dates,
            y: data.vwap,
            type: 'scatter',
            mode: 'lines',
            name: 'VWAP',
            line: { color: '#fd7e14', width: 2 }
        },
        {
            x: data.dates,
            y: data.close,
            type: 'scatter',
            mode: 'lines',
            name: 'Close',
            line: { color: '#0d6efd', width: 2 }
        }
    ];

    const layout = {
        ...defaultLayout,
        xaxis: { ...defaultLayout.xaxis, title: 'Date' },
        yaxis: { ...defaultLayout.yaxis, title: 'Price (USD)' }
    };

    Plotly.newPlot('vwap-chart', traces, layout, defaultConfig);
}