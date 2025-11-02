from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, time, timedelta
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
import numpy as np
import re
import time as time_module  # Added to help with ThreadPoolExecutor issues
import argparse
from lumibot.brokers import Binance  # Added Binance broker import

# Alpaca credentials
API_KEY = "PKLHX78JAR8WGPJ4BG79" 
API_SECRET = "0uew90OcJLdUkEmW5QVC1SXc57BZxgSxKp8a8nVs" 
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True,
    "WEBSOCKET_URL": "wss://paper-api.alpaca.markets/stream"
}

# Binance credentials
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET"

BINANCE_CREDS = {
    "API_KEY": BINANCE_API_KEY,
    "API_SECRET": BINANCE_API_SECRET,
    "PAPER": True,  # Ensures paper trading
    "TESTNET": True,  # Uses Binance testnet
    "BASE_URL": "https://testnet.binance.vision"  # Binance testnet API endpoint
}

def test_news_keywords(api_key, api_secret, base_url, lookback_days=5, news_keywords=None, geopolitical_keywords=None):
    """
    Test function to check if we can find relevant news matching our keywords.
    Returns the found headlines and statistics about keyword matches.
    """
    # Initialize default keywords if not provided
    news_keywords = news_keywords or [
        "gold", "precious metal", "inflation", "federal reserve", 
        "interest rate", "dollar", "safe haven", "treasury yield",
        "central bank", "recession", "economic uncertainty", "rate hike",
        "rate cut", "cpi", "consumer price", "jobs report", "unemployment"
    ]
    
    geopolitical_keywords = geopolitical_keywords or [
        "war", "conflict", "sanctions", "tension", "crisis", "attack",
        "missile", "nuclear", "military", "terrorism", "coup", "unrest",
        "protest", "diplomatic", "treaty", "agreement", "dispute"
    ]
    
    # Initialize API
    api = REST(base_url=base_url, key_id=api_key, secret_key=api_secret)
    
    # Get date range
    today = datetime.now()
    days_prior = today - timedelta(days=lookback_days)
    today_str = today.strftime('%Y-%m-%d')
    days_prior_str = days_prior.strftime('%Y-%m-%d')
    
    print(f"Searching for news from {days_prior_str} to {today_str}")
    
    try:
        # Get general market news
        print("Fetching market news...")
        market_news = api.get_news(start=days_prior_str, end=today_str)
        time_module.sleep(0.5)  # Prevent thread pool issues
        
        market_headlines = []
        for ev in market_news:
            try:
                market_headlines.append(ev.__dict__["_raw"]["headline"])
            except (KeyError, AttributeError):
                continue
        
        print(f"Found {len(market_headlines)} general market headlines")
        
        # Try to get gold-specific news
        gold_headlines = []
        try:
            print("Fetching gold-specific news...")
            gold_news = api.get_news(symbol="GLD", start=days_prior_str, end=today_str)
            time_module.sleep(0.5)
            
            for ev in gold_news:
                try:
                    gold_headlines.append(ev.__dict__["_raw"]["headline"])
                except (KeyError, AttributeError):
                    continue
            
            print(f"Found {len(gold_headlines)} gold-specific headlines")
        except Exception as e:
            print(f"Could not get gold-specific news: {str(e)}")
        
        # Combine all headlines
        all_headlines = market_headlines + gold_headlines
        print(f"Total headlines found: {len(all_headlines)}")
        
        # Filter for relevant headlines
        relevant_news = []
        geo_news = []
        
        for headline in all_headlines:
            headline_lower = headline.lower()
            
            # Check for financial relevance
            if any(keyword.lower() in headline_lower for keyword in news_keywords):
                relevant_news.append(("FINANCIAL", headline))
            
            # Check for geopolitical factors
            if any(keyword.lower() in headline_lower for keyword in geopolitical_keywords):
                geo_news.append(("GEOPOLITICAL", headline))
        
        # Combine unique headlines
        all_relevant = []
        for category, headline in relevant_news:
            if not any(h == headline for _, h in all_relevant):
                all_relevant.append((category, headline))
                
        for category, headline in geo_news:
            if not any(h == headline for _, h in all_relevant):
                all_relevant.append((category, headline))
        
        # Count matches per keyword
        keyword_matches = {}
        for keyword in news_keywords:
            count = sum(1 for _, headline in all_relevant if keyword.lower() in headline.lower())
            if count > 0:
                keyword_matches[keyword] = count
                
        for keyword in geopolitical_keywords:
            count = sum(1 for _, headline in all_relevant if keyword.lower() in headline.lower())
            if count > 0:
                keyword_matches[keyword] = count
        
        # Print results
        print(f"\nFound {len(all_relevant)} relevant headlines:")
        for category, headline in all_relevant:
            print(f"[{category}] {headline}")
        
        print("\nKeyword match statistics:")
        for keyword, count in sorted(keyword_matches.items(), key=lambda x: x[1], reverse=True):
            print(f"- '{keyword}': {count} matches")
        
        if not all_relevant:
            print("\nNo relevant news found matching your keywords.")
            print("You might want to expand your keyword list or check if the API is returning news correctly.")
        
        return all_relevant, keyword_matches
        
    except Exception as e:
        print(f"Error testing news keywords: {str(e)}")
        return [], {}

class EnhancedGoldSentimentTrader(Strategy): 
    def initialize(self, 
                   symbol:str="GLD",  # GLD ETF for gold
                   binance_symbol:str="XAUUSDT",  # Binance symbol for Gold/USD
                   cash_at_risk:float=0.5,
                   sentiment_threshold:float=0.95,
                   lookback_days:int=5,
                   take_profit_pct:float=0.05,
                   stop_loss_pct:float=0.03,
                   sentiment_confirmation_count:int=2,
                   news_keywords:list=None,
                   geopolitical_keywords:list=None,
                   usd_symbol:str="UUP",  # ETF tracking US Dollar
                   binance_usd_symbol:str="USDTUSDC",  # Binance USD equivalent
                   min_volume_ratio:float=1.2,  # Minimum volume compared to average
                   broker_type:str="alpaca"):  # Added broker type parameter
        
        self.symbol = symbol
        self.binance_symbol = binance_symbol
        self.broker_type = broker_type
        
        # Use appropriate symbol based on broker
        self.trading_symbol = self.binance_symbol if self.broker_type == "binance" else self.symbol
        self.usd_symbol = binance_usd_symbol if self.broker_type == "binance" else usd_symbol
        
        self.sleeptime = "1H"  # Check hourly for more responsive trading
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.sentiment_threshold = sentiment_threshold
        self.lookback_days = lookback_days
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_volume_ratio = min_volume_ratio
        
        # Sentiment confirmation system
        self.sentiment_confirmation_count = sentiment_confirmation_count
        self.recent_sentiments = []  # Store recent sentiment readings
        
        # Keywords to filter relevant news for gold
        self.news_keywords = news_keywords or [
            "gold", "precious metal", "inflation", "federal reserve", 
            "interest rate", "dollar", "safe haven", "treasury yield",
            "central bank", "recession", "economic uncertainty", "rate hike",
            "rate cut", "cpi", "consumer price", "jobs report", "unemployment"
        ]
        
        # Specific keywords for geopolitical events (which often boost gold)
        self.geopolitical_keywords = geopolitical_keywords or [
            "war", "conflict", "sanctions", "tension", "crisis", "attack",
            "missile", "nuclear", "military", "terrorism", "coup", "unrest",
            "protest", "diplomatic", "treaty", "agreement", "dispute"
        ]
        
        # Initialize REST API for news (only for Alpaca)
        if self.broker_type == "alpaca":
            self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        else:
            self.api = None  # Will need alternative news source for Binance
            
        self.sentiment_history = []  # Track sentiment over time
        self.trading_session_data = {}  # Store data about trading sessions
        
        # Initialize sentiment confirmation tracking
        for _ in range(sentiment_confirmation_count):
            self.recent_sentiments.append(("neutral", 0))

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.trading_symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        
        # For Binance, ensure quantity meets minimum requirements
        if self.broker_type == "binance":
            # This is a simplified approach - in production you'd check the actual asset's minimum quantity
            min_quantity = 0.001  # Minimum quantity for many Binance assets
            quantity = max(quantity, min_quantity)
            # Round to appropriate decimal places for the asset
            quantity = round(quantity, 3)  # Adjust based on your asset's precision
            
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        days_prior = today - Timedelta(days=self.lookback_days)
        return today.strftime('%Y-%m-%d'), days_prior.strftime('%Y-%m-%d')

    def check_trading_window(self):
        """Check if current time is within optimal trading windows for gold"""
        current_time = self.get_datetime().time()
        current_day = self.get_datetime().weekday()
        
        # For Binance, trading is 24/7, but we'll still check for optimal times
        if self.broker_type == "binance":
            # Skip lowest liquidity periods on weekends
            if current_day in [5, 6] and current_time >= time(21, 0) and current_time <= time(4, 0):
                return False, "Weekend low liquidity period - avoiding trade"
            # Always allow trading during other times (24/7)
            return True, "Binance 24/7 trading - proceeding with caution"
        
        # For Alpaca/traditional markets:
        # Skip trading on weekends (5=Saturday, 6=Sunday)
        if current_day in [5, 6]:
            return False, "Weekend - no trading"
            
        # Check for NY/London overlap (8AM-12PM ET) - prime gold trading time
        if current_time >= time(8, 0) and current_time <= time(12, 0):
            return True, "NY/London overlap - optimal trading window"
            
        # Check for post major economic releases (typically 8:30AM, 10:00AM ET)
        # Allow trading 1-2 hours after these times
        if (current_time >= time(9, 30) and current_time <= time(11, 0)) or \
           (current_time >= time(11, 0) and current_time <= time(12, 30)):
            return True, "Post economic release window - good trading time"
            
        # Asian session for Fed announcement follow-through (7PM-2AM ET)
        if current_time >= time(19, 0) or current_time <= time(2, 0):
            # Check if there was a Fed announcement today or yesterday
            if self.check_fed_announcement():
                return True, "Asian session after Fed announcement - good for follow-through moves"
                
        return False, "Not in optimal trading window"

    def check_fed_announcement(self):
        """Check if there was a recent Fed announcement"""
        # This is a simplified implementation
        # In a real system, you would check a calendar API or database
        today = self.get_datetime()
        
        # For Binance, we may need an alternative source for Fed news
        if self.broker_type == "binance" or self.api is None:
            # Simplified implementation for Binance - could be replaced with a free news API
            self.log_message("Warning: Fed announcement check using fallback method for Binance")
            return False  # Placeholder - implement alternative news source
        
        # For Alpaca, use the existing code
        # Check news for Fed-related keywords in recent headlines
        fed_terms = ["fomc", "fed ", "federal reserve", "powell", "rate decision"]
        
        try:
            # Using a try/except since API calls might fail
            recent_news = self.api.get_news(
                start=(today - Timedelta(days=1)).strftime('%Y-%m-%d'),
                end=today.strftime('%Y-%m-%d')
            )
            
            # Add a small delay to prevent overwhelming the thread pool
            time_module.sleep(0.1)
            
            headlines = [ev.__dict__["_raw"]["headline"].lower() for ev in recent_news]
            
            # Check if any headlines contain Fed terms
            return any(any(term in headline for term in fed_terms) for headline in headlines)
        except Exception as e:
            self.log_message(f"Error checking Fed announcements: {str(e)}")
            return False

    def check_volume_confirmation(self):
        """Check if trading volume is sufficiently high to confirm market interest"""
        try:
            # Get current volume
            current_bars = self.get_historical_prices(self.trading_symbol, 1, "day")
            if not current_bars or len(current_bars) == 0:
                return False, "Unable to get current volume data"
                
            current_volume = current_bars[0].volume
            
            # Get average volume over last 20 days
            historical_bars = self.get_historical_prices(self.trading_symbol, 20, "day")
            if not historical_bars or len(historical_bars) < 10:
                return True, "Insufficient historical volume data, proceeding anyway"
                
            avg_volume = sum(bar.volume for bar in historical_bars) / len(historical_bars)
            
            # Check if current volume is significantly higher than average
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio >= self.min_volume_ratio:
                return True, f"Volume confirmation: current {current_volume} is {volume_ratio:.2f}x average"
            else:
                return False, f"Insufficient volume: current {current_volume} is only {volume_ratio:.2f}x average"
                
        except Exception as e:
            self.log_message(f"Error checking volume: {str(e)}")
            return False, f"Error checking volume: {str(e)}"

    def check_usd_alignment(self, sentiment):
        """Check if USD movement aligns with expected gold movement"""
        try:
            # Binance might need different symbols or approach
            if self.broker_type == "binance" and not self.usd_symbol:
                return True, "USD alignment check not available for Binance, proceeding anyway"
                
            # Get USD performance
            usd_bars = self.get_historical_prices(self.usd_symbol, 2, "day")
            if not usd_bars or len(usd_bars) < 2:
                return True, "Unable to get USD data, proceeding anyway"
                
            # Calculate daily percent change for USD
            usd_prev_close = usd_bars[0].close
            usd_current = self.get_last_price(self.usd_symbol)
            usd_change = (usd_current - usd_prev_close) / usd_prev_close * 100
            
            # Check for inverse correlation with sentiment
            if sentiment == "positive" and usd_change < -0.2:
                return True, f"USD down {usd_change:.2f}% aligns with positive gold sentiment"
            elif sentiment == "negative" and usd_change > 0.2:
                return True, f"USD up {usd_change:.2f}% aligns with negative gold sentiment"
            else:
                return False, f"USD movement ({usd_change:.2f}%) doesn't strongly confirm {sentiment} gold sentiment"
                
        except Exception as e:
            self.log_message(f"Error checking USD alignment: {str(e)}")
            return True, f"Error checking USD alignment: {str(e)}, proceeding anyway"

    def filter_relevant_news(self, headlines):
        """Filter headlines that are relevant to gold trading"""
        relevant_news = []
        geopolitical_news = []
        
        for headline in headlines:
            headline_lower = headline.lower()
            
            # Check for gold-specific relevance
            if any(keyword.lower() in headline_lower for keyword in self.news_keywords):
                relevant_news.append(headline)
            
            # Check for geopolitical factors
            if any(keyword.lower() in headline_lower for keyword in self.geopolitical_keywords):
                geopolitical_news.append(headline)
        
        # Combine both types of news, but mark geopolitical ones
        combined_news = relevant_news.copy()
        for headline in geopolitical_news:
            if headline not in combined_news:
                combined_news.append(headline)
        
        # If no relevant news, return original headlines
        return combined_news if combined_news else headlines, len(geopolitical_news) > 0

    def get_sentiment(self): 
        today, days_prior = self.get_dates()
        
        # For Binance, we need an alternative news source
        if self.broker_type == "binance" or self.api is None:
            # This is a placeholder for an alternative news API implementation
            self.log_message("Using alternative news source for Binance (placeholder)")
            
            # In production, implement a call to a free financial news API here
            # For now, we'll just return neutral sentiment to allow the strategy to continue
            dummy_probability = 0.5
            dummy_sentiment = "neutral"
            
            # Update sentiment history for tracking purposes
            self.sentiment_history.append((self.get_datetime(), dummy_sentiment, dummy_probability))
            
            # Update recent sentiments for confirmation tracking
            self.recent_sentiments.append((dummy_sentiment, dummy_probability))
            if len(self.recent_sentiments) > self.sentiment_confirmation_count:
                self.recent_sentiments.pop(0)
                
            return dummy_probability, dummy_sentiment, False
        
        # For Alpaca, use the existing implementation
        try:
            # Get general market news - use try/except to handle potential errors
            market_news = self.api.get_news(start=days_prior, end=today)
            # Add small delay to prevent overwhelming thread pool
            time_module.sleep(0.1)
            
            market_headlines = [ev.__dict__["_raw"]["headline"] for ev in market_news]
            
            # Get symbol-specific news (if available)
            try:
                symbol_news = self.api.get_news(symbol=self.symbol, start=days_prior, end=today)
                # Add small delay
                time_module.sleep(0.1)
                
                symbol_headlines = [ev.__dict__["_raw"]["headline"] for ev in symbol_news]
                all_headlines = market_headlines + symbol_headlines
            except Exception as e:
                self.log_message(f"Could not get symbol-specific news: {str(e)}")
                all_headlines = market_headlines
                
            # Filter for relevant headlines
            relevant_headlines, has_geopolitical = self.filter_relevant_news(all_headlines)
            
            if not relevant_headlines:
                self.log_message(f"No relevant news found for {self.symbol}")
                return 0, "neutral", False
                
            # Get sentiment analysis
            probability, sentiment = estimate_sentiment(relevant_headlines)
            
            # Boost sentiment probability slightly if geopolitical news is present
            # This reflects gold's traditional role as a geopolitical safe haven
            if has_geopolitical and sentiment == "positive":
                probability = min(float(probability) * 1.05, 1.0)
                self.log_message("Sentiment boosted due to geopolitical news")
            
            # Log sentiment information
            self.log_message(f"Sentiment: {sentiment}, Probability: {probability:.4f}, Headlines: {len(relevant_headlines)}")
            
            # Track sentiment over time
            self.sentiment_history.append((self.get_datetime(), sentiment, float(probability)))
            
            # Update recent sentiments for confirmation tracking
            self.recent_sentiments.append((sentiment, float(probability)))
            if len(self.recent_sentiments) > self.sentiment_confirmation_count:
                self.recent_sentiments.pop(0)
            
            # Check if we have sufficient confirmation of sentiment
            sentiment_confirmed = self.check_sentiment_confirmation(sentiment)
            
            return probability, sentiment, sentiment_confirmed
            
        except Exception as e:
            self.log_message(f"Error getting sentiment: {str(e)}")
            return 0, "neutral", False

    def check_sentiment_confirmation(self, current_sentiment):
        """Check if sentiment has been consistent enough to act on"""
        # Count how many recent sentiments match the current one
        matching_count = sum(1 for s, p in self.recent_sentiments 
                             if s == current_sentiment and p >= self.sentiment_threshold)
        
        # Need majority confirmation
        threshold = self.sentiment_confirmation_count // 2 + 1
        
        return matching_count >= threshold

    def calculate_dynamic_targets(self, sentiment_probability, sentiment, has_geopolitical=False):
        """Dynamically adjust profit targets based on sentiment strength and market conditions"""
        # Base values
        take_profit_multiplier = self.take_profit_pct
        stop_loss_multiplier = self.stop_loss_pct
        
        # Adjust based on sentiment strength
        sentiment_strength = float(sentiment_probability) - self.sentiment_threshold
        if sentiment_strength > 0:
            # Scale up profit target and down stop loss as sentiment gets stronger
            take_profit_multiplier += sentiment_strength * 0.1  # Max +10% adjustment
            stop_loss_multiplier -= sentiment_strength * 0.01   # Max -1% adjustment
        
        # Adjust for geopolitical factors (which can cause larger moves in gold)
        if has_geopolitical and sentiment == "positive":
            take_profit_multiplier *= 1.2  # Increase profit target by 20%
            stop_loss_multiplier *= 0.9   # Tighten stop loss by 10%
            
        # Check current market volatility (via ATR or similar measure)
        try:
            volatility_adjustment = self.check_volatility()
            take_profit_multiplier *= volatility_adjustment
            stop_loss_multiplier *= volatility_adjustment
        except:
            pass
            
        return take_profit_multiplier, stop_loss_multiplier

    def check_volatility(self):
        """Calculate relative market volatility for position sizing"""
        try:
            # Get recent price bars
            bars = self.get_historical_prices(self.trading_symbol, 14, "day")
            
            if not bars or len(bars) < 14:
                return 1.0  # Default no adjustment
                
            # Simple volatility measure: (high-low) / close
            volatilities = [(bar.high - bar.low) / bar.close for bar in bars]
            avg_volatility = sum(volatilities) / len(volatilities)
            
            # Get current day's volatility
            current_volatility = (bars[0].high - bars[0].low) / bars[0].close
            
            # Calculate relative volatility
            relative_volatility = current_volatility / avg_volatility
            
            # Limit adjustment factor between 0.8 and 1.5
            return max(0.8, min(relative_volatility, 1.5))
            
        except Exception as e:
            self.log_message(f"Error calculating volatility: {str(e)}")
            return 1.0  # Default no adjustment

    def on_trading_iteration(self):
        try:
            # First check if we're in a good trading window
            in_trading_window, window_message = self.check_trading_window()
            if not in_trading_window:
                self.log_message(f"Not trading: {window_message}")
                return
                
            self.log_message(f"Trading window active: {window_message}")
            
            # Get current market conditions
            cash, last_price, quantity = self.position_sizing() 
            probability, sentiment, sentiment_confirmed = self.get_sentiment()
            
            # Skip if not enough cash or neutral sentiment
            if cash <= last_price or sentiment == "neutral" or float(probability) < self.sentiment_threshold:
                return
                
            # Skip if sentiment isn't confirmed
            if not sentiment_confirmed:
                self.log_message(f"Skipping trade: {sentiment} sentiment not confirmed yet")
                return
                
            # Check volume for confirmation
            volume_confirmed, volume_message = self.check_volume_confirmation()
            if not volume_confirmed:
                self.log_message(f"Skipping trade: {volume_message}")
                return
                
            self.log_message(volume_message)
            
            # Check USD alignment
            usd_aligned, usd_message = self.check_usd_alignment(sentiment)
            if not usd_aligned:
                self.log_message(f"Caution: {usd_message}")
                # We proceed but with reduced position size
                quantity = int(quantity * 0.7)
                
            # Calculate dynamic profit/loss targets
            take_profit_pct, stop_loss_pct = self.calculate_dynamic_targets(
                probability, sentiment, has_geopolitical=any("geopolitical" in msg for msg in self.get_logs())
            )
            
            # All checks passed, proceed with trade
            if sentiment == "positive" and float(probability) >= self.sentiment_threshold: 
                # Strong positive sentiment for gold - typically during uncertainty, inflation concerns
                if self.last_trade == "sell": 
                    self.sell_all() 
                    
                order = self.create_order(
                    self.trading_symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price * (1 + take_profit_pct), 
                    stop_loss_price=last_price * (1 - stop_loss_pct)
                )
                self.submit_order(order) 
                self.last_trade = "buy"
                self.log_message(f"BUY order: {quantity} shares at ${last_price:.2f} | TP: ${last_price * (1 + take_profit_pct):.2f} | SL: ${last_price * (1 - stop_loss_pct):.2f}")
                
            elif sentiment == "negative" and float(probability) >= self.sentiment_threshold: 
                # Strong negative sentiment for gold - typically during economic strength, rising rates
                if self.last_trade == "buy": 
                    self.sell_all() 
                    
                order = self.create_order(
                    self.trading_symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price * (1 - take_profit_pct), 
                    stop_loss_price=last_price * (1 + stop_loss_pct)
                )
                self.submit_order(order) 
                self.last_trade = "sell"
                self.log_message(f"SELL order: {quantity} shares at ${last_price:.2f} | TP: ${last_price * (1 - take_profit_pct):.2f} | SL: ${last_price * (1 + stop_loss_pct):.2f}")
        
        except Exception as e:
            self.log_message(f"Error in trading iteration: {str(e)}")

    def teardown(self):
        """Analyze performance after backtest"""
        try:
            # Calculate sentiment accuracy
            if hasattr(self, 'sentiment_history') and self.sentiment_history:
                self.log_message("Sentiment Analysis Summary:")
                pos_count = sum(1 for _, s, _ in self.sentiment_history if s == "positive")
                neg_count = sum(1 for _, s, _ in self.sentiment_history if s == "negative")
                neu_count = sum(1 for _, s, _ in self.sentiment_history if s == "neutral")
                
                self.log_message(f"Positive signals: {pos_count}, Negative signals: {neg_count}, Neutral: {neu_count}")
                
                # Trade timing analysis
                if hasattr(self, 'orders'):
                    morning_trades = sum(1 for o in self.orders if o.created_at.time() >= time(8,0) and o.created_at.time() <= time(12,0))
                    afternoon_trades = sum(1 for o in self.orders if o.created_at.time() > time(12,0) and o.created_at.time() <= time(16,0))
                    evening_trades = sum(1 for o in self.orders if o.created_at.time() > time(16,0) or o.created_at.time() < time(8,0))
                    
                    self.log_message(f"Trading session analysis: Morning: {morning_trades}, Afternoon: {afternoon_trades}, Evening: {evening_trades}")
        except Exception as e:
            self.log_message(f"Error in teardown: {str(e)}")
            
    def test_connection():
        try:
            # Test Alpaca connection
            alpaca_api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
            account = alpaca_api.get_account()
            print(f"Alpaca connection successful! Account ID: {account.id}")
            
            # Alternatively, try a simple request
            import requests
            response = requests.get("https://paper-api.alpaca.markets/v2/account", 
                                headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET})
            print(f"Alpaca HTTP status: {response.status_code}")
            return True
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Gold Sentiment Trading Bot')
    parser.add_argument('--mode', type=str, choices=['live', 'test_news'], default='test_news',
                        help='Mode to run: "live" for live trading, "test_news" to test news keywords')
    parser.add_argument('--broker', type=str, choices=['alpaca', 'binance'], default='alpaca',
                        help='Broker to use: "alpaca" or "binance"')
    parser.add_argument('--lookback', type=int, default=5,
                        help='Number of days to look back for news (default: 5)')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Trading symbol (default: GLD for Alpaca, XAUUSDT for Binance)')