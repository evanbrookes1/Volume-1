from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
import pandas as pd
from serpapi import GoogleSearch
import os
from datetime import datetime, timedelta
from functools import lru_cache
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- Initialize the Flask App and API Clients ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- IMPORTANT: Paste your API Keys here ---
FRED_API_KEY = 'cb01340f23187af8b7aa9ad74332224b'
SERPAPI_API_KEY = 'df91d1155080c329a6e06a991486bd9d965e3e73c4df204178dd189ee47e2c38'

fred = Fred(api_key=FRED_API_KEY)




# =====================================================================
# === PORTFOLIO OPTIMIZER SECTION =====================================
# =====================================================================

def get_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculates return, volatility, and Sharpe ratio for given weights."""
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev > 0 else 0
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Function to minimize for Max Sharpe Ratio portfolio."""
    return -get_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Function to get the portfolio volatility."""
    return get_portfolio_performance(weights, mean_returns, cov_matrix, 0)[1]

def portfolio_return(weights, mean_returns, cov_matrix, risk_free_rate):
    """Function to get the portfolio return."""
    return get_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[0]

def neg_sortino_ratio(weights, mean_returns, daily_returns, risk_free_rate):
    """Function to minimize for Max Sortino Ratio portfolio."""
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
    annualized_return = np.mean(portfolio_daily_returns) * 252
    target_return = risk_free_rate / 252
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < target_return]
    downside_deviation = np.std(downside_returns) * np.sqrt(252)
    if downside_deviation == 0: return 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    return -sortino_ratio

def calculate_cvar(weights, daily_returns, alpha=0.95):
    """Function to minimize for Min CVaR portfolio."""
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
    var = np.percentile(portfolio_daily_returns, (1 - alpha) * 100)
    cvar = portfolio_daily_returns[portfolio_daily_returns <= var].mean()
    return -cvar

# =====================================================================
# === NEW: Advanced Metrics Calculation Functions =====================
# =====================================================================

def calculate_drawdown(cumulative_returns_series):
    """Calculates the historical drawdown series and the maximum drawdown."""
    # A drawdown is a peak-to-trough decline during a specific period.
    running_max = cumulative_returns_series.cummax()
    drawdown = (cumulative_returns_series - running_max) / running_max
    return drawdown, drawdown.min()

def calculate_calmar_ratio(annualized_return, max_drawdown):
    """Calculates the Calmar Ratio."""
    # The Calmar ratio uses max drawdown in its calculation.
    # We use abs() because max_drawdown is negative.
    return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

def calculate_beta(portfolio_daily_returns, market_daily_returns):
    """Calculates the portfolio's beta against the market."""
    # Beta measures the volatility of an investment compared to the market as a whole.
    covariance_matrix = np.cov(portfolio_daily_returns, market_daily_returns)
    covariance = covariance_matrix[0, 1]
    market_variance = np.var(market_daily_returns)
    return covariance / market_variance if market_variance != 0 else 1 # Default to 1 if market has no variance

def calculate_jensens_alpha(portfolio_annual_return, market_annual_return, risk_free_rate, beta):
    """Calculates Jensen's Alpha."""
    # Jensen's Alpha represents the "abnormal return" of a security or portfolio
    # over the theoretical expected return (calculated by CAPM).
    expected_return = risk_free_rate + beta * (market_annual_return - risk_free_rate)
    return portfolio_annual_return - expected_return


def find_optimal_portfolio(optimizer_func, args, num_assets, constraints=None):
    """A generic function to find an optimal portfolio using an optimizer."""
    if constraints is None:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    result = minimize(optimizer_func, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=50):
    """Generate efficient frontier points more robustly."""
    num_assets = len(mean_returns)
    min_vol_result = find_optimal_portfolio(portfolio_volatility, (mean_returns, cov_matrix), num_assets)
    min_vol_return = portfolio_return(min_vol_result.x, mean_returns, cov_matrix, risk_free_rate)
    max_return_idx = np.argmax(mean_returns)
    max_return_weights = np.zeros(num_assets)
    max_return_weights[max_return_idx] = 1.0
    max_return_val = portfolio_return(max_return_weights, mean_returns, cov_matrix, risk_free_rate)
    
    target_returns = np.linspace(min_vol_return, max_return_val, num_points)
    efficient_portfolios = []
    
    for target_ret in target_returns:
        try:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns, cov_matrix, risk_free_rate) - target_ret}
            ]
            result = find_optimal_portfolio(portfolio_volatility, (mean_returns, cov_matrix), num_assets, constraints)
            if result.success:
                vol = result.fun
                sharpe = (target_ret - risk_free_rate) / vol if vol > 0 else 0
                efficient_portfolios.append([target_ret, vol, sharpe])
        except Exception as e:
            continue
    return efficient_portfolios

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    """Generates a list of random portfolios with their performance metrics."""
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        p_return, p_vol, p_sharpe = get_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0,i] = p_return
        results[1,i] = p_vol
        results[2,i] = p_sharpe
    return results.T.tolist()


# =====================================================================
# === PORTFOLIO OPTIMIZER API ENDPOINT ================================
# =====================================================================

@app.route('/optimize-portfolio', methods=['POST'])
def optimize_portfolio():
    try:
        post_data = request.get_json()
        tickers = post_data.get('tickers')
        
        if not tickers or len(tickers) < 2:
            return jsonify({"error": "Please provide at least two tickers."}), 400

        # --- Data Fetching and Cleaning ---
        all_tickers = tickers + ['SPY']
        data = yf.download(all_tickers, period='1y', progress=False)
        if data.empty:
            return jsonify({"error": "Could not fetch data for the provided tickers."}), 400
        
        close_prices = data['Close'].dropna(axis='columns', how='all')
        valid_tickers = [ticker for ticker in tickers if ticker in close_prices.columns]
        
        if len(valid_tickers) < 2:
            return jsonify({"error": f"Could not retrieve valid historical data for at least two tickers."}), 400
        
        if 'SPY' not in close_prices.columns:
             return jsonify({"error": "Could not retrieve market data (SPY). Please try again."}), 400

        cleaned_prices = close_prices[valid_tickers + ['SPY']].dropna()
        returns = cleaned_prices.pct_change().dropna()

        if len(returns) < 2:
            return jsonify({"error": "Not enough overlapping historical data for a calculation."}), 400

        spy_returns = returns['SPY']
        portfolio_returns = returns.drop(columns=['SPY'])
        
        mean_returns = portfolio_returns.mean()
        cov_matrix = portfolio_returns.cov()
        num_assets = len(valid_tickers)
        correlation_matrix = portfolio_returns.corr().to_dict()

        try:
            ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
            risk_free_rate = ten_year_treasury_rate.iloc[-1]
        except Exception:
            risk_free_rate = 0.02 # Fallback risk-free rate

        # --- Find Key Portfolios ---
        max_sharpe_result = find_optimal_portfolio(neg_sharpe_ratio, (mean_returns, cov_matrix, risk_free_rate), num_assets)
        max_sharpe_weights = max_sharpe_result.x
        max_sharpe_perf = get_portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)

        min_vol_result = find_optimal_portfolio(portfolio_volatility, (mean_returns, cov_matrix), num_assets)
        min_vol_weights = min_vol_result.x
        min_vol_perf = get_portfolio_performance(min_vol_weights, mean_returns, cov_matrix, risk_free_rate)

        max_sortino_result = find_optimal_portfolio(neg_sortino_ratio, (mean_returns, portfolio_returns, risk_free_rate), num_assets)
        max_sortino_weights = max_sortino_result.x
        max_sortino_perf = get_portfolio_performance(max_sortino_weights, mean_returns, cov_matrix, risk_free_rate)
        sortino_ratio_val = -neg_sortino_ratio(max_sortino_weights, mean_returns, portfolio_returns, risk_free_rate)

        min_cvar_result = find_optimal_portfolio(calculate_cvar, (portfolio_returns, 0.95), num_assets)
        min_cvar_weights = min_cvar_result.x
        min_cvar_perf = get_portfolio_performance(min_cvar_weights, mean_returns, cov_matrix, risk_free_rate)
        cvar_val = -calculate_cvar(min_cvar_weights, portfolio_returns, 0.95)

        # --- Generate Chart Data ---
        efficient_frontier_points = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=50)
        random_portfolios = generate_random_portfolios(5000, mean_returns, cov_matrix, risk_free_rate)

        # --- Calculate Individual Asset Performance ---
        individual_asset_performance = []
        annual_returns = mean_returns * 252
        annual_volatilities = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
        for i, ticker in enumerate(valid_tickers):
            individual_asset_performance.append({"ticker": ticker, "return": annual_returns[i], "volatility": annual_volatilities[i]})

        # --- Calculate Historical Performance for chart ---
        max_sharpe_daily_returns = (portfolio_returns * max_sharpe_weights).sum(axis=1)
        min_var_daily_returns = (portfolio_returns * min_vol_weights).sum(axis=1)
        cumulative_max_sharpe_returns = (1 + max_sharpe_daily_returns).cumprod()
        cumulative_min_var_returns = (1 + min_var_daily_returns).cumprod()
        cumulative_spy_returns = (1 + spy_returns).cumprod()
        
        # =====================================================================
        # === NEW: Calculate Advanced Metrics for Key Portfolios ==============
        # =====================================================================
        advanced_metrics = {}
        market_annual_return = spy_returns.mean() * 252
        
        # --- Metrics for Max Sharpe Portfolio ---
        dd_series_sharpe, max_dd_sharpe = calculate_drawdown(cumulative_max_sharpe_returns)
        calmar_sharpe = calculate_calmar_ratio(max_sharpe_perf[0], max_dd_sharpe)
        beta_sharpe = calculate_beta(max_sharpe_daily_returns, spy_returns)
        alpha_sharpe = calculate_jensens_alpha(max_sharpe_perf[0], market_annual_return, risk_free_rate, beta_sharpe)
        
        advanced_metrics['maxSharpe'] = {
            "maxDrawdown": max_dd_sharpe,
            "calmarRatio": calmar_sharpe,
            "beta": beta_sharpe,
            "jensensAlpha": alpha_sharpe,
            "drawdownSeries": {"dates": [d.strftime('%Y-%m-%d') for d in dd_series_sharpe.index], "values": dd_series_sharpe.tolist()}
        }
        
        # --- Metrics for Min Variance Portfolio ---
        dd_series_min_var, max_dd_min_var = calculate_drawdown(cumulative_min_var_returns)
        calmar_min_var = calculate_calmar_ratio(min_vol_perf[0], max_dd_min_var)
        beta_min_var = calculate_beta(min_var_daily_returns, spy_returns)
        alpha_min_var = calculate_jensens_alpha(min_vol_perf[0], market_annual_return, risk_free_rate, beta_min_var)
        
        advanced_metrics['minVariance'] = {
            "maxDrawdown": max_dd_min_var,
            "calmarRatio": calmar_min_var,
            "beta": beta_min_var,
            "jensensAlpha": alpha_min_var,
            "drawdownSeries": {"dates": [d.strftime('%Y-%m-%d') for d in dd_series_min_var.index], "values": dd_series_min_var.tolist()}
        }


        # --- Prepare final JSON response for the frontend ---
        response_data = {
            "minVariance": {
                "weights": [{"ticker": ticker, "weight": weight} for ticker, weight in zip(valid_tickers, min_vol_weights)],
                "returns": min_vol_perf[0], "volatility": min_vol_perf[1], "sharpeRatio": min_vol_perf[2]
            },
            "maxSharpe": {
                "weights": [{"ticker": ticker, "weight": weight} for ticker, weight in zip(valid_tickers, max_sharpe_weights)],
                "returns": max_sharpe_perf[0], "volatility": max_sharpe_perf[1], "sharpeRatio": max_sharpe_perf[2]
            },
            "maxSortino": {
                "weights": [{"ticker": ticker, "weight": weight} for ticker, weight in zip(valid_tickers, max_sortino_weights)],
                "returns": max_sortino_perf[0], "volatility": max_sortino_perf[1], "sharpeRatio": max_sortino_perf[2],
                "sortinoRatio": sortino_ratio_val
            },
            "minCVaR": {
                "weights": [{"ticker": ticker, "weight": weight} for ticker, weight in zip(valid_tickers, min_cvar_weights)],
                "returns": min_cvar_perf[0], "volatility": min_cvar_perf[1], "sharpeRatio": min_cvar_perf[2],
                "cvar": cvar_val
            },
            "efficientFrontier": efficient_frontier_points,
            "randomPortfolios": random_portfolios,
            "individualAssets": individual_asset_performance,
            "riskFreeRate": risk_free_rate,
            "performance": {
                "dates": [date.strftime('%Y-%m-%d') for date in cumulative_max_sharpe_returns.index],
                "maxSharpe": cumulative_max_sharpe_returns.tolist(),
                "minVariance": cumulative_min_var_returns.tolist(),
                "sp500": cumulative_spy_returns.tolist()
            },
            "correlationMatrix": correlation_matrix,
            "advancedMetrics": advanced_metrics # <-- NEW: Added advanced metrics here
        }
        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred. " + str(e)}), 500


# =====================================================================
# === FINANCIAL NEWS SECTION ==========================================
# =====================================================================

# --- News Fetching API Endpoint ---
@app.route('/get-financial-news', methods=['GET'])
def get_financial_news():
    try:
        if SERPAPI_API_KEY == "YOUR_SERPAPI_KEY_HERE":
            return jsonify({"error": "SerpApi API key not configured in app.py."}), 500

        params = {
            "engine": "google_news",
            "q": "uk finance business market",
            "api_key": SERPAPI_API_KEY,
            "num": 10
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "news_results" not in results:
            return jsonify({"error": "Could not fetch news results."}), 500

        articles = []
        for item in results["news_results"]:
            source_info = item.get("source")
            source_name = ""
            if isinstance(source_info, dict):
                source_name = source_info.get("name")
            elif isinstance(source_info, str):
                source_name = source_info
            
            title = item.get("title", "No Title Available")
            snippet = item.get("snippet")
            if not snippet:
                snippet = title[:150] + "..." if len(title) > 150 else title

            articles.append({
                "title": title,
                "link": item.get("link"),
                "source": source_name,
                "snippet": snippet,
                "date": item.get("date"),
                "thumbnail": item.get("thumbnail")
            })

        return jsonify(articles)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# Market overview section   
@app.route('/get-market-snapshot', methods=['GET'])
def get_market_snapshot():
    try:
        market_tickers = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "FTSE 100": "^FTSE",
            "Gold": "GC=F",
            "Oil (Crude)": "CL=F",
            "Bitcoin": "BTC-USD"
        }
        
        snapshot_data = []
        for name, ticker in market_tickers.items():
            info = yf.Ticker(ticker).info
            current_price = info.get('regularMarketPrice', info.get('previousClose'))
            previous_close = info.get('previousClose')
            
            if current_price and previous_close:
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100
                snapshot_data.append({
                    "name": name,
                    "price": f"{current_price:,.2f}",
                    "change": f"{change:+.2f}",
                    "percent_change": f"{percent_change:+.2f}%",
                    "is_positive": change >= 0
                })
        
        return jsonify(snapshot_data)

    except Exception as e:
        print(f"An unexpected error occurred in get_market_snapshot: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-economic-calendar', methods=['GET'])
def get_economic_calendar():
    try:
        # --- MOCK DATA IMPLEMENTATION ---
        # In a real application, you would fetch this from a dedicated financial data API.
        # yfinance.get_calendar() is often limited 
        today = datetime.now()
        events = []
        # Generate sample events for the next 7 days
        for i in range(7):
            event_date = today + timedelta(days=i)
            if event_date.weekday() < 5: # Monday to Friday
                events.extend([
                    {"date": event_date.strftime('%Y-%m-%d'), "time": "09:30", "country": "GB", "currency": "GBP", "event": "Manufacturing PMI", "impact": "high", "actual": "51.4", "forecast": "51.3", "previous": "51.2"},
                    {"date": event_date.strftime('%Y-%m-%d'), "time": "10:00", "country": "EU", "currency": "EUR", "event": "CPI Flash Estimate y/y", "impact": "high", "actual": "2.5%", "forecast": "2.5%", "previous": "2.4%"},
                    {"date": event_date.strftime('%Y-%m-%d'), "time": "13:30", "country": "US", "currency": "USD", "event": "Non-Farm Employment Change", "impact": "high", "actual": "210K", "forecast": "185K", "previous": "175K"},
                    {"date": event_date.strftime('%Y-%m-%d'), "time": "15:00", "country": "CA", "currency": "CAD", "event": "Ivey PMI", "impact": "medium", "actual": "64.1", "forecast": "65.2", "previous": "63.0"},
                    {"date": event_date.strftime('%Y-%m-%d'), "time": "23:50", "country": "JP", "currency": "JPY", "event": "BoJ Core CPI y/y", "impact": "low", "actual": "2.1%", "forecast": "2.2%", "previous": "2.2%"}
                ])
        
        # Group events by date for easier frontend processing
        df = pd.DataFrame(events)
        if not df.empty:
            grouped_events = df.groupby('date').apply(lambda x: x.to_dict('records')).reset_index(name='events').to_dict('records')
        else:
            grouped_events = []
            
        return jsonify(grouped_events)

    except Exception as e:
        print(f"An unexpected error occurred in get_economic_calendar: {e}")
        return jsonify({"error": str(e)}), 500



# =====================================================================
# === TRADE SIGNALS SECTION ===========================================
# =====================================================================

# =====================================================================
# === MARKET SCREENER SECTION =========================================
# =====================================================================

@lru_cache(maxsize=1)
def get_all_tickers():
    """
    Fetch all available stock tickers from multiple sources.
    This function is cached to avoid repeated API calls.
    """
    all_tickers = set()
    
    try:
        # Method 1: Get S&P 500 tickers
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_table['Symbol'].tolist()
        all_tickers.update(sp500_tickers)
        
        # Method 2: Get NASDAQ tickers
        nasdaq_url = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt'
        nasdaq_response = requests.get(nasdaq_url)
        if nasdaq_response.status_code == 200:
            nasdaq_tickers = nasdaq_response.text.strip().split('\n')
            all_tickers.update(nasdaq_tickers)
        
        # Method 3: Get NYSE tickers
        nyse_url = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt'
        nyse_response = requests.get(nyse_url)
        if nyse_response.status_code == 200:
            nyse_tickers = nyse_response.text.strip().split('\n')
            all_tickers.update(nyse_tickers)
        
        # Method 4: Get popular ETFs
        etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'EFA', 'EEM', 'GLD', 'SLV', 
                'USO', 'UNG', 'TLT', 'IEF', 'LQD', 'HYG', 'AGG', 'BND', 'VNQ', 'XLRE',
                'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'ARKK']
        all_tickers.update(etfs)
        
        # Clean the tickers
        all_tickers = {ticker.strip().upper() for ticker in all_tickers if ticker and len(ticker) <= 5}
        
        return list(all_tickers)
    
    except Exception as e:
        print(f"Error fetching all tickers: {e}")
        # Return a default list if fetching fails
        return get_default_tickers()

def get_default_tickers():
    """Return a default list of popular tickers as fallback"""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM', 'CVX', 'ABBV', 'PFE', 'KO',
            'WMT', 'PEP', 'TMO', 'CSCO', 'MRK', 'VZ', 'ABT', 'ADBE', 'CMCSA', 'NKE', 'INTC',
            'CRM', 'ACN', 'NFLX', 'AMD', 'LIN', 'TXN', 'DHR', 'PM', 'WFC', 'NEE', 'RTX',
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'GLD', 'SLV', 'ARKK', 'XLF']

def fetch_stock_data_for_screener(ticker, period='1mo'):
    """Fetch comprehensive data for a single stock with error handling"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Skip if no valid info
        if not info or 'regularMarketPrice' not in info:
            return None
        
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
        
        # Get current price
        current_price = info.get('regularMarketPrice', info.get('previousClose', 0))
        
        if not current_price or current_price <= 0:
            return None
        
        # Calculate price changes
        price_1d_ago = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_1w_ago = hist['Close'].iloc[-5] if len(hist) > 5 else current_price
        price_1m_ago = hist['Close'].iloc[0] if len(hist) > 0 else current_price
        
        change_1d = ((current_price - price_1d_ago) / price_1d_ago * 100) if price_1d_ago else 0
        change_1w = ((current_price - price_1w_ago) / price_1w_ago * 100) if price_1w_ago else 0
        change_1m = ((current_price - price_1m_ago) / price_1m_ago * 100) if price_1m_ago else 0
        
        # Moving averages
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else current_price
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else current_price
        
        # RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50
        
        # Volume analysis
        avg_volume_20d = hist['Volume'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else 0
        current_volume = hist['Volume'].iloc[-1] if len(hist) > 0 else 0
        volume_ratio = (current_volume / avg_volume_20d) if avg_volume_20d > 0 else 1
        
        # 52-week high/low
        if len(hist) >= 252:
            high_52w = hist['High'].tail(252).max()
            low_52w = hist['Low'].tail(252).min()
        else:
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
        
        pct_from_52w_high = ((current_price - high_52w) / high_52w * 100) if high_52w else 0
        pct_from_52w_low = ((current_price - low_52w) / low_52w * 100) if low_52w else 0
        
        # Volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'current_price': round(current_price, 2),
            'change_1d': round(change_1d, 2),
            'change_1w': round(change_1w, 2),
            'change_1m': round(change_1m, 2),
            'volume': int(current_volume),
            'avg_volume_20d': int(avg_volume_20d),
            'volume_ratio': round(volume_ratio, 2),
            'pe_ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 0,
            'forward_pe': round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else 0,
            'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0,
            'profit_margin': round(info.get('profitMargins', 0) * 100, 2) if info.get('profitMargins') else 0,
            'rsi': round(current_rsi, 2),
            'price_vs_sma20': round(((current_price - sma_20) / sma_20 * 100), 2) if sma_20 else 0,
            'price_vs_sma50': round(((current_price - sma_50) / sma_50 * 100), 2) if sma_50 else 0,
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'pct_from_52w_high': round(pct_from_52w_high, 2),
            'pct_from_52w_low': round(pct_from_52w_low, 2),
            'volatility': round(volatility, 2),
            'beta': round(info.get('beta', 1), 2) if info.get('beta') else 1,
            'eps': round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 0,
            'revenue_growth': round(info.get('revenueGrowth', 0) * 100, 2) if info.get('revenueGrowth') else 0,
            'recommendation': info.get('recommendationKey', 'none')
        }
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

@app.route('/screener/run', methods=['POST'])
def run_screener():
    """Run market screener with custom filters on all available stocks"""
    try:
        data = request.get_json()
        
        # Get screening parameters
        filters = data.get('filters', {})
        sort_by = data.get('sort_by', 'market_cap')
        sort_order = data.get('sort_order', 'desc')
        limit = min(data.get('limit', 100), 500)  # Cap at 500 for performance
        use_all_stocks = data.get('use_all_stocks', False)
        custom_tickers = data.get('custom_tickers', [])
        
        # Get tickers to screen
        if custom_tickers:
            tickers = custom_tickers
        elif use_all_stocks:
            # For initial screening, use a subset for performance
            all_tickers = get_all_tickers()
            # Pre-filter based on basic criteria if screening all stocks
            if 'min_market_cap' in filters and filters['min_market_cap'] > 1000000000:
                # For large market cap requirements, focus on major exchanges
                tickers = all_tickers[:1000]  # Limit to first 1000 for performance
            else:
                tickers = all_tickers[:500]  # Limit initial scan
        else:
            # Default to popular stocks
            tickers = get_default_tickers()
        
        print(f"Screening {len(tickers)} stocks...")
        
        # Fetch data for all stocks in parallel with progress tracking
        stock_data = []
        batch_size = 50  # Process in batches to avoid overwhelming the API
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_stock_data_for_screener, ticker): ticker 
                          for ticker in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        stock_data.append(result)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(tickers):
                time.sleep(0.5)
        
        print(f"Successfully fetched data for {len(stock_data)} stocks")
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(stock_data)
        
        if df.empty:
            return jsonify({
                "summary": {
                    "total_screened": 0,
                    "results_count": 0,
                    "filters_applied": filters
                },
                "results": []
            })
        
        # Apply filters
        filtered_df = apply_screener_filters(df, filters)
        
        # Sort results
        if sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_order == 'asc'))
        
        # Limit results
        filtered_df = filtered_df.head(limit)
        
        # Convert to dict and clean NaN values
        results = filtered_df.replace({np.nan: None}).to_dict('records')
        
        # Add screening summary
        summary = {
            'total_screened': len(stock_data),
            'results_count': len(results),
            'filters_applied': filters,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'summary': summary,
            'results': results
        })
        
    except Exception as e:
        print(f"Screener error: {e}")
        return jsonify({"error": str(e)}), 500

def apply_screener_filters(df, filters):
    """Apply screening filters to the DataFrame"""
    
    # Price filters
    if 'min_price' in filters and filters['min_price'] is not None:
        df = df[df['current_price'] >= filters['min_price']]
    if 'max_price' in filters and filters['max_price'] is not None:
        df = df[df['current_price'] <= filters['max_price']]
    
    # Market cap filters
    if 'min_market_cap' in filters and filters['min_market_cap'] is not None:
        df = df[df['market_cap'] >= filters['min_market_cap']]
    if 'max_market_cap' in filters and filters['max_market_cap'] is not None:
        df = df[df['market_cap'] <= filters['max_market_cap']]
    
    # PE ratio filters
    if 'min_pe' in filters and filters['min_pe'] is not None:
        df = df[(df['pe_ratio'] >= filters['min_pe']) & (df['pe_ratio'] > 0)]
    if 'max_pe' in filters and filters['max_pe'] is not None:
        df = df[df['pe_ratio'] <= filters['max_pe']]
    
    # Volume filters
    if 'min_volume' in filters and filters['min_volume'] is not None:
        df = df[df['volume'] >= filters['min_volume']]
    if 'min_avg_volume' in filters and filters['min_avg_volume'] is not None:
        df = df[df['avg_volume_20d'] >= filters['min_avg_volume']]
    if 'min_volume_ratio' in filters and filters['min_volume_ratio'] is not None:
        df = df[df['volume_ratio'] >= filters['min_volume_ratio']]
    
    # Performance filters
    if 'min_change_1d' in filters and filters['min_change_1d'] is not None:
        df = df[df['change_1d'] >= filters['min_change_1d']]
    if 'max_change_1d' in filters and filters['max_change_1d'] is not None:
        df = df[df['change_1d'] <= filters['max_change_1d']]
    if 'min_change_1w' in filters and filters['min_change_1w'] is not None:
        df = df[df['change_1w'] >= filters['min_change_1w']]
    if 'min_change_1m' in filters and filters['min_change_1m'] is not None:
        df = df[df['change_1m'] >= filters['min_change_1m']]
    
    # Technical filters
    if 'min_rsi' in filters and filters['min_rsi'] is not None:
        df = df[df['rsi'] >= filters['min_rsi']]
    if 'max_rsi' in filters and filters['max_rsi'] is not None:
        df = df[df['rsi'] <= filters['max_rsi']]
    if 'above_sma20' in filters and filters['above_sma20']:
        df = df[df['price_vs_sma20'] > 0]
    if 'above_sma50' in filters and filters['above_sma50']:
        df = df[df['price_vs_sma50'] > 0]
    if 'below_sma20' in filters and filters['below_sma20']:
        df = df[df['price_vs_sma20'] < 0]
    if 'below_sma50' in filters and filters['below_sma50']:
        df = df[df['price_vs_sma50'] < 0]
    
    # 52-week range filters
    if 'near_52w_high' in filters and filters['near_52w_high']:
        df = df[df['pct_from_52w_high'] >= -10]  # Within 10% of 52-week high
    if 'near_52w_low' in filters and filters['near_52w_low']:
        df = df[df['pct_from_52w_low'] <= 10]  # Within 10% of 52-week low
    
    # Fundamental filters
    if 'min_dividend_yield' in filters and filters['min_dividend_yield'] is not None:
        df = df[df['dividend_yield'] >= filters['min_dividend_yield']]
    if 'min_profit_margin' in filters and filters['min_profit_margin'] is not None:
        df = df[df['profit_margin'] >= filters['min_profit_margin']]
    
    # Volatility filters
    if 'max_volatility' in filters and filters['max_volatility'] is not None:
        df = df[df['volatility'] <= filters['max_volatility']]
    if 'min_beta' in filters and filters['min_beta'] is not None:
        df = df[df['beta'] >= filters['min_beta']]
    if 'max_beta' in filters and filters['max_beta'] is not None:
        df = df[df['beta'] <= filters['max_beta']]
    
    # Sector filter
    if 'sectors' in filters and filters['sectors']:
        df = df[df['sector'].isin(filters['sectors'])]
    
    # Remove stocks with invalid data
    df = df[df['current_price'] > 0]
    df = df[df['market_cap'] > 0]
    
    return df

@app.route('/screener/presets', methods=['GET'])
def get_screener_presets():
    """Get predefined screening presets"""
    presets = {
        'oversold': {
            'name': 'Oversold Stocks',
            'description': 'Stocks that may be oversold and due for a bounce',
            'filters': {
                'max_rsi': 30,
                'min_market_cap': 1000000000,
                'min_avg_volume': 500000
            }
        },
        'overbought': {
            'name': 'Overbought Stocks',
            'description': 'Stocks that may be overbought',
            'filters': {
                'min_rsi': 70,
                'min_market_cap': 1000000000,
                'min_avg_volume': 500000
            }
        },
        'momentum': {
            'name': 'Momentum Stocks',
            'description': 'Stocks showing strong upward momentum',
            'filters': {
                'min_change_1m': 10,
                'above_sma20': True,
                'above_sma50': True,
                'min_volume_ratio': 1.5,
                'min_market_cap': 1000000000
            }
        },
        'value': {
            'name': 'Value Stocks',
            'description': 'Potentially undervalued stocks',
            'filters': {
                'max_pe': 15,
                'min_pe': 0,
                'min_dividend_yield': 2,
                'min_market_cap': 10000000000
            }
        },
        'growth': {
            'name': 'Growth Stocks',
            'description': 'High growth potential stocks',
            'filters': {
                'min_revenue_growth': 20,
                'min_market_cap': 1000000000
            }
        },
        'day_trading': {
            'name': 'Day Trading Candidates',
            'description': 'High volume, volatile stocks for day trading',
            'filters': {
                'min_volume': 5000000,
                'min_change_1d': 2,
                'min_volatility': 30
            }
        },
        'penny_stocks': {
            'name': 'Active Penny Stocks',
            'description': 'Low-priced stocks with high volume',
            'filters': {
                'max_price': 5,
                'min_volume': 1000000
            }
        },
        'large_cap_stable': {
            'name': 'Large Cap Stable',
            'description': 'Large, stable companies',
            'filters': {
                'min_market_cap': 50000000000,
                'max_volatility': 25,
                'min_profit_margin': 5
            }
        },
        'breakout': {
            'name': 'Breakout Candidates',
            'description': 'Stocks breaking through resistance',
            'filters': {
                'near_52w_high': True,
                'min_volume_ratio': 2,
                'min_change_1d': 3
            }
        },
        'reversal': {
            'name': 'Reversal Candidates',
            'description': 'Stocks that might reverse from lows',
            'filters': {
                'near_52w_low': True,
                'max_rsi': 35,
                'min_market_cap': 1000000000
            }
        }
    }
    
    return jsonify(presets)

@app.route('/screener/search-ticker', methods=['GET'])
def search_ticker():
    """Search for ticker symbols by company name"""
    try:
        query = request.args.get('q', '').upper()
        if not query:
            return jsonify([])
        
        all_tickers = get_all_tickers()
        
        # Simple search - match tickers that start with query
        matches = [t for t in all_tickers if t.startswith(query)][:20]
        
        return jsonify(matches)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/screener/sectors', methods=['GET'])
def get_sectors():
    """Get list of available sectors"""
    sectors = [
        'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
        'Communication Services', 'Industrials', 'Consumer Defensive',
        'Energy', 'Real Estate', 'Basic Materials', 'Utilities'
    ]
    return jsonify({'sectors': sectors})
#-------------------------------------------------------------------------------------------------------------------------
def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line"""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(data, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic Oscillator"""
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    
    k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    k_percent = k_percent.rolling(window=smooth_k).mean()
    d_percent = k_percent.rolling(window=smooth_d).mean()
    
    return k_percent, d_percent

def calculate_atr(data, period=14):
    """Calculate Average True Range for volatility"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def calculate_volume_indicators(data):
    """Calculate volume-based indicators"""
    # On-Balance Volume (OBV)
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    
    # Volume Moving Average
    volume_sma = data['Volume'].rolling(window=20).mean()
    
    # Volume Rate of Change
    volume_roc = ((data['Volume'] - data['Volume'].shift(10)) / data['Volume'].shift(10)) * 100
    
    return obv, volume_sma, volume_roc

def get_support_resistance(data, window=20):
    """Calculate basic support and resistance levels"""
    recent_data = data.tail(window)
    support = recent_data['Low'].min()
    resistance = recent_data['High'].max()
    pivot = (recent_data['High'].iloc[-1] + recent_data['Low'].iloc[-1] + recent_data['Close'].iloc[-1]) / 3
    
    return support, resistance, pivot

def calculate_advanced_indicators(data):
    """Calculate additional technical indicators for comprehensive analysis"""
    indicators = {}
    
    # Price-based indicators
    indicators['SMA_20'] = data['Close'].rolling(window=20).mean()
    indicators['SMA_50'] = data['Close'].rolling(window=50).mean()
    indicators['SMA_200'] = data['Close'].rolling(window=200).mean()
    indicators['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    indicators['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    indicators['RSI_14'] = calculate_rsi(data, 14)
    indicators['RSI_9'] = calculate_rsi(data, 9)
    
    # MACD
    indicators['MACD'], indicators['MACD_signal'], indicators['MACD_histogram'] = calculate_macd(data)
    
    # Bollinger Bands
    indicators['BB_upper'], indicators['BB_middle'], indicators['BB_lower'] = calculate_bollinger_bands(data)
    
    # Stochastic
    indicators['Stoch_K'], indicators['Stoch_D'] = calculate_stochastic(data)
    
    # ATR for volatility
    indicators['ATR'] = calculate_atr(data)
    
    # Volume indicators
    indicators['OBV'], indicators['Volume_SMA'], indicators['Volume_ROC'] = calculate_volume_indicators(data)
    
    # Support and Resistance
    indicators['Support'], indicators['Resistance'], indicators['Pivot'] = get_support_resistance(data)
    
    return indicators

def generate_signal_from_indicator(indicator_name, current_value, data, indicators):
    """Generate trading signal based on specific indicator"""
    signal = "HOLD"
    confidence = 0.5
    
    if 'RSI' in indicator_name:
        if current_value > 70:
            signal = "SELL"
            confidence = min((current_value - 70) / 30, 1.0)
        elif current_value < 30:
            signal = "BUY"
            confidence = min((30 - current_value) / 30, 1.0)
    
    elif indicator_name == 'MACD':
        if indicators['MACD'].iloc[-1] > indicators['MACD_signal'].iloc[-1]:
            signal = "BUY"
            confidence = 0.7
        elif indicators['MACD'].iloc[-1] < indicators['MACD_signal'].iloc[-1]:
            signal = "SELL"
            confidence = 0.7
    
    elif 'SMA' in indicator_name or 'EMA' in indicator_name:
        if 'SMA_50' in indicators and 'SMA_200' in indicators:
            if not np.isnan(indicators['SMA_50'].iloc[-1]) and not np.isnan(indicators['SMA_200'].iloc[-1]):
                if indicators['SMA_50'].iloc[-1] > indicators['SMA_200'].iloc[-1]:
                    signal = "BUY"
                    confidence = 0.8
                elif indicators['SMA_50'].iloc[-1] < indicators['SMA_200'].iloc[-1]:
                    signal = "SELL"
                    confidence = 0.8
    
    elif indicator_name == 'Bollinger_Bands':
        current_price = data['Close'].iloc[-1]
        if current_price > indicators['BB_upper'].iloc[-1]:
            signal = "SELL"
            confidence = 0.6
        elif current_price < indicators['BB_lower'].iloc[-1]:
            signal = "BUY"
            confidence = 0.6
    
    elif indicator_name == 'Stochastic':
        k_value = indicators['Stoch_K'].iloc[-1]
        d_value = indicators['Stoch_D'].iloc[-1]
        if k_value > 80:
            signal = "SELL"
            confidence = 0.6
        elif k_value < 20:
            signal = "BUY"
            confidence = 0.6
        elif k_value > d_value and k_value < 50:
            signal = "BUY"
            confidence = 0.5
        elif k_value < d_value and k_value > 50:
            signal = "SELL"
            confidence = 0.5
    
    elif indicator_name == 'Volume':
        current_volume = data['Volume'].iloc[-1]
        avg_volume = indicators['Volume_SMA'].iloc[-1]
        if current_volume > avg_volume * 1.5:
            # High volume could confirm current trend
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            if price_change > 0:
                signal = "BUY"
                confidence = 0.6
            elif price_change < 0:
                signal = "SELL"
                confidence = 0.6
    
    return signal, confidence

@app.route('/get-trade-signals', methods=['POST'])
def get_trade_signals():
    """Enhanced endpoint for trading signals with multiple indicators"""
    try:
        post_data = request.get_json()
        tickers = post_data.get('tickers')
        timeframe = post_data.get('timeframe', 'medium')  # short, medium, long
        
        if not tickers:
            return jsonify({"error": "No tickers provided."}), 400

        # Adjust data period based on timeframe
        period_map = {
            'short': '1mo',
            'medium': '3mo',
            'long': '1y'
        }
        period = period_map.get(timeframe, '3mo')
        
        all_signals_data = []

        for ticker in tickers:
            try:
                # Fetch data with OHLCV
                data = yf.download(ticker, period=period, progress=False)
                
                if data.empty:
                    print(f"Could not fetch data for {ticker}, skipping.")
                    continue

                # Flatten columns if MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Calculate all indicators
                indicators = calculate_advanced_indicators(data)
                
                # Generate signals for each indicator group
                indicator_signals = []
                
                # 1. Moving Average Crossover
                ma_signal = "HOLD"
                ma_confidence = 0.5
                if not np.isnan(indicators['SMA_50'].iloc[-1]) and not np.isnan(indicators['SMA_200'].iloc[-1]):
                    if indicators['SMA_50'].iloc[-1] > indicators['SMA_200'].iloc[-1]:
                        ma_signal = "BUY"
                        ma_confidence = 0.8
                    elif indicators['SMA_50'].iloc[-1] < indicators['SMA_200'].iloc[-1]:
                        ma_signal = "SELL"
                        ma_confidence = 0.8
                indicator_signals.append({
                    "name": "Moving Average (50/200)",
                    "signal": ma_signal,
                    "confidence": ma_confidence,
                    "value": f"50MA: ${indicators['SMA_50'].iloc[-1]:.2f}" if not np.isnan(indicators['SMA_50'].iloc[-1]) else "N/A"
                })
                
                # 2. RSI Signal
                rsi_signal = "HOLD"
                rsi_14 = indicators['RSI_14'].iloc[-1]
                rsi_confidence = 0.5
                if not np.isnan(rsi_14):
                    if rsi_14 > 70:
                        rsi_signal = "SELL"
                        rsi_confidence = min((rsi_14 - 70) / 30, 1.0)
                    elif rsi_14 < 30:
                        rsi_signal = "BUY"
                        rsi_confidence = min((30 - rsi_14) / 30, 1.0)
                indicator_signals.append({
                    "name": "RSI (14)",
                    "signal": rsi_signal,
                    "confidence": rsi_confidence,
                    "value": f"{rsi_14:.1f}" if not np.isnan(rsi_14) else "N/A"
                })
                
                # 3. MACD Signal
                macd_signal = "HOLD"
                macd_confidence = 0.5
                if not np.isnan(indicators['MACD'].iloc[-1]) and not np.isnan(indicators['MACD_signal'].iloc[-1]):
                    if indicators['MACD'].iloc[-1] > indicators['MACD_signal'].iloc[-1]:
                        macd_signal = "BUY"
                        macd_confidence = 0.7
                    elif indicators['MACD'].iloc[-1] < indicators['MACD_signal'].iloc[-1]:
                        macd_signal = "SELL"
                        macd_confidence = 0.7
                indicator_signals.append({
                    "name": "MACD",
                    "signal": macd_signal,
                    "confidence": macd_confidence,
                    "value": f"Histogram: {indicators['MACD_histogram'].iloc[-1]:.3f}" if not np.isnan(indicators['MACD_histogram'].iloc[-1]) else "N/A"
                })
                
                # 4. Bollinger Bands Signal
                bb_signal = "HOLD"
                bb_confidence = 0.5
                current_price = data['Close'].iloc[-1]
                if not np.isnan(indicators['BB_upper'].iloc[-1]) and not np.isnan(indicators['BB_lower'].iloc[-1]):
                    if current_price > indicators['BB_upper'].iloc[-1]:
                        bb_signal = "SELL"
                        bb_confidence = 0.6
                    elif current_price < indicators['BB_lower'].iloc[-1]:
                        bb_signal = "BUY"
                        bb_confidence = 0.6
                    
                    # Calculate position within bands
                    bb_width = indicators['BB_upper'].iloc[-1] - indicators['BB_lower'].iloc[-1]
                    bb_position = (current_price - indicators['BB_lower'].iloc[-1]) / bb_width if bb_width > 0 else 0.5
                    
                indicator_signals.append({
                    "name": "Bollinger Bands",
                    "signal": bb_signal,
                    "confidence": bb_confidence,
                    "value": f"Position: {bb_position:.1%}" if not np.isnan(indicators['BB_upper'].iloc[-1]) else "N/A"
                })
                
                # 5. Stochastic Signal
                stoch_signal = "HOLD"
                stoch_confidence = 0.5
                if not np.isnan(indicators['Stoch_K'].iloc[-1]):
                    k_value = indicators['Stoch_K'].iloc[-1]
                    d_value = indicators['Stoch_D'].iloc[-1]
                    if k_value > 80:
                        stoch_signal = "SELL"
                        stoch_confidence = 0.6
                    elif k_value < 20:
                        stoch_signal = "BUY"
                        stoch_confidence = 0.6
                indicator_signals.append({
                    "name": "Stochastic",
                    "signal": stoch_signal,
                    "confidence": stoch_confidence,
                    "value": f"%K: {k_value:.1f}" if not np.isnan(k_value) else "N/A"
                })
                
                # 6. Volume Analysis
                vol_signal = "HOLD"
                vol_confidence = 0.5
                if not np.isnan(indicators['Volume_SMA'].iloc[-1]):
                    current_volume = data['Volume'].iloc[-1]
                    avg_volume = indicators['Volume_SMA'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio > 1.5:
                        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                        if price_change > 0:
                            vol_signal = "BUY"
                            vol_confidence = 0.6
                        elif price_change < 0:
                            vol_signal = "SELL"
                            vol_confidence = 0.6
                            
                indicator_signals.append({
                    "name": "Volume Analysis",
                    "signal": vol_signal,
                    "confidence": vol_confidence,
                    "value": f"Vol Ratio: {volume_ratio:.1f}x" if not np.isnan(indicators['Volume_SMA'].iloc[-1]) else "N/A"
                })
                
                # Add price info and additional metrics
                current_price = data['Close'].iloc[-1]
                price_change_1d = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
                price_change_5d = ((current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100 if len(data) > 5 else 0
                
                # Calculate overall signal strength
                buy_count = sum(1 for sig in indicator_signals if sig['signal'] == 'BUY')
                sell_count = sum(1 for sig in indicator_signals if sig['signal'] == 'SELL')
                total_confidence = sum(sig['confidence'] for sig in indicator_signals) / len(indicator_signals)
                
                stock_data = {
                    "ticker": ticker,
                    "current_price": f"${current_price:.2f}",
                    "price_change_1d": f"{price_change_1d:+.2f}%",
                    "price_change_5d": f"{price_change_5d:+.2f}%",
                    "support": f"${indicators['Support']:.2f}",
                    "resistance": f"${indicators['Resistance']:.2f}",
                    "volatility": f"{indicators['ATR'].iloc[-1]:.2f}" if not np.isnan(indicators['ATR'].iloc[-1]) else "N/A",
                    "indicators": indicator_signals,
                    "summary": {
                        "buy_signals": buy_count,
                        "sell_signals": sell_count,
                        "hold_signals": len(indicator_signals) - buy_count - sell_count,
                        "overall_confidence": f"{total_confidence:.1%}"
                    }
                }
                
                all_signals_data.append(stock_data)
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue

        return jsonify(all_signals_data)
        
    except Exception as e:
        print(f"An unexpected error occurred in get_trade_signals: {e}")
        return jsonify({"error": str(e)}), 500



# --- Start the Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)