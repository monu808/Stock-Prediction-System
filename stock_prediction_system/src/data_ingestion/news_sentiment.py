"""News and sentiment data ingestion from various Indian financial news sources"""

import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..utils.logging_config import LoggingMixin


@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_label: str
    symbols_mentioned: List[str]
    category: str
    importance_score: float = 0.0


class EconomicTimesProvider(LoggingMixin):
    """Economic Times news provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = "https://economictimes.indiatimes.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    async def initialize(self) -> None:
        """Initialize Economic Times provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        self.logger.info("Economic Times provider initialized")
    
    async def fetch_latest_news(self, limit: int = 50) -> List[NewsArticle]:
        """Fetch latest news articles"""
        try:
            # Markets section
            markets_url = f"{self.base_url}/markets"
            articles = []
            
            async with self.session.get(markets_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find article links
                    article_links = soup.find_all('a', href=True)
                    processed_urls = set()
                    
                    for link in article_links[:limit]:
                        href = link.get('href')
                        if href and '/markets/' in href and href not in processed_urls:
                            processed_urls.add(href)
                            
                            # Make URL absolute
                            if href.startswith('/'):
                                full_url = f"{self.base_url}{href}"
                            else:
                                full_url = href
                            
                            article = await self._fetch_article_content(full_url)
                            if article:
                                articles.append(article)
                            
                            # Rate limiting
                            await asyncio.sleep(0.5)
            
            self.logger.info(f"Fetched {len(articles)} articles from Economic Times")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Economic Times news: {e}")
            return []
    
    async def _fetch_article_content(self, url: str) -> Optional[NewsArticle]:
        """Fetch individual article content"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_elem = soup.find('h1') or soup.find('title')
                    title = title_elem.get_text().strip() if title_elem else ""
                    
                    # Extract content
                    content_div = soup.find('div', class_='artText') or soup.find('div', class_='Normal')
                    content = ""
                    if content_div:
                        paragraphs = content_div.find_all('p')
                        content = ' '.join([p.get_text().strip() for p in paragraphs])
                    
                    if title and content:
                        # Analyze sentiment
                        sentiment = self._analyze_sentiment(f"{title} {content}")
                        
                        # Extract mentioned symbols
                        symbols = self._extract_symbols(f"{title} {content}")
                        
                        return NewsArticle(
                            title=title,
                            content=content,
                            source="Economic Times",
                            url=url,
                            published_at=datetime.now(),
                            sentiment_score=sentiment['compound'],
                            sentiment_label=self._get_sentiment_label(sentiment['compound']),
                            symbols_mentioned=symbols,
                            category="markets",
                            importance_score=self._calculate_importance(title, content, symbols)
                        )
        except Exception as e:
            self.logger.error(f"Error fetching article {url}: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # VADER sentiment analysis
        return self.sentiment_analyzer.polarity_scores(text)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label from score"""
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        # Common Indian stock symbols and company names
        symbols_map = {
            'reliance': 'RELIANCE',
            'tcs': 'TCS',
            'infosys': 'INFY',
            'hdfc': 'HDFCBANK',
            'icici': 'ICICIBANK',
            'sbi': 'SBIN',
            'bharti': 'BHARTIARTL',
            'itc': 'ITC',
            'wipro': 'WIPRO',
            'maruti': 'MARUTI',
            'asian paints': 'ASIANPAINT',
            'bajaj': 'BAJFINANCE',
            'kotak': 'KOTAKBANK',
            'axis': 'AXISBANK',
            'adani': 'ADANIPORTS',
            'tata': 'TATAMOTORS',
            'larsen': 'LT',
            'hindustan unilever': 'HINDUNILVR',
            'ntpc': 'NTPC',
            'ongc': 'ONGC'
        }
        
        mentioned_symbols = []
        text_lower = text.lower()
        
        for company, symbol in symbols_map.items():
            if company in text_lower:
                mentioned_symbols.append(symbol)
        
        return list(set(mentioned_symbols))
    
    def _calculate_importance(self, title: str, content: str, symbols: List[str]) -> float:
        """Calculate importance score of the article"""
        score = 0.0
        
        # Base score
        score += 0.3
        
        # Title keywords
        important_keywords = [
            'breaking', 'exclusive', 'alert', 'surge', 'crash', 'rally',
            'results', 'earnings', 'profit', 'loss', 'dividend', 'split',
            'merger', 'acquisition', 'ipo', 'delisting', 'bonus'
        ]
        
        title_lower = title.lower()
        for keyword in important_keywords:
            if keyword in title_lower:
                score += 0.1
        
        # Symbols mentioned
        score += len(symbols) * 0.1
        
        # Content length (longer articles might be more important)
        if len(content) > 500:
            score += 0.1
        
        return min(score, 1.0)
    
    async def close(self) -> None:
        """Close Economic Times provider"""
        if self.session:
            await self.session.close()
        self.logger.info("Economic Times provider closed")


class MoneyControlProvider(LoggingMixin):
    """MoneyControl news provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = "https://www.moneycontrol.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    async def initialize(self) -> None:
        """Initialize MoneyControl provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        self.logger.info("MoneyControl provider initialized")
    
    async def fetch_latest_news(self, limit: int = 50) -> List[NewsArticle]:
        """Fetch latest news from MoneyControl"""
        try:
            articles = []
            news_url = f"{self.base_url}/news/business/markets/"
            
            async with self.session.get(news_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find news articles
                    article_links = soup.find_all('a', href=True)
                    processed_urls = set()
                    
                    for link in article_links[:limit]:
                        href = link.get('href')
                        if href and '/news/' in href and href not in processed_urls:
                            processed_urls.add(href)
                            
                            if href.startswith('/'):
                                full_url = f"{self.base_url}{href}"
                            else:
                                full_url = href
                            
                            article = await self._fetch_article_content(full_url)
                            if article:
                                articles.append(article)
                            
                            await asyncio.sleep(0.5)
            
            self.logger.info(f"Fetched {len(articles)} articles from MoneyControl")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching MoneyControl news: {e}")
            return []
    
    async def _fetch_article_content(self, url: str) -> Optional[NewsArticle]:
        """Fetch individual article content from MoneyControl"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_elem = soup.find('h1', class_='article_title') or soup.find('h1')
                    title = title_elem.get_text().strip() if title_elem else ""
                    
                    # Extract content
                    content_div = soup.find('div', class_='content_wrapper') or soup.find('div', class_='arti-flow')
                    content = ""
                    if content_div:
                        paragraphs = content_div.find_all('p')
                        content = ' '.join([p.get_text().strip() for p in paragraphs])
                    
                    if title and content:
                        sentiment = self._analyze_sentiment(f"{title} {content}")
                        symbols = self._extract_symbols(f"{title} {content}")
                        
                        return NewsArticle(
                            title=title,
                            content=content,
                            source="MoneyControl",
                            url=url,
                            published_at=datetime.now(),
                            sentiment_score=sentiment['compound'],
                            sentiment_label=self._get_sentiment_label(sentiment['compound']),
                            symbols_mentioned=symbols,
                            category="markets",
                            importance_score=self._calculate_importance(title, content, symbols)
                        )
        except Exception as e:
            self.logger.error(f"Error fetching MoneyControl article {url}: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        return SentimentIntensityAnalyzer().polarity_scores(text)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label"""
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols - same logic as Economic Times"""
        symbols_map = {
            'reliance': 'RELIANCE',
            'tcs': 'TCS',
            'infosys': 'INFY',
            'hdfc': 'HDFCBANK',
            # Add more mappings...
        }
        
        mentioned_symbols = []
        text_lower = text.lower()
        
        for company, symbol in symbols_map.items():
            if company in text_lower:
                mentioned_symbols.append(symbol)
        
        return list(set(mentioned_symbols))
    
    def _calculate_importance(self, title: str, content: str, symbols: List[str]) -> float:
        """Calculate article importance"""
        # Same logic as Economic Times
        return 0.5  # Simplified for now
    
    async def close(self) -> None:
        """Close MoneyControl provider"""
        if self.session:
            await self.session.close()
        self.logger.info("MoneyControl provider closed")


class NewsAggregator(LoggingMixin):
    """Main news aggregator orchestrating multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.is_running = False
        self.news_callbacks = []
        
        # Initialize providers
        news_sources = config.get('news_sources', {})
        
        if news_sources.get('economic_times', {}).get('enabled', False):
            self.providers['et'] = EconomicTimesProvider(news_sources['economic_times'])
        
        if news_sources.get('moneycontrol', {}).get('enabled', False):
            self.providers['mc'] = MoneyControlProvider(news_sources['moneycontrol'])
    
    async def initialize(self) -> None:
        """Initialize all news providers"""
        for name, provider in self.providers.items():
            await provider.initialize()
            self.logger.info(f"Initialized {name} news provider")
    
    async def start_news_collection(self, interval: int = 300) -> None:
        """Start news collection (every 5 minutes by default)"""
        self.is_running = True
        self.logger.info("Starting news collection")
        
        while self.is_running:
            # Collect news from all providers
            all_articles = []
            
            for provider_name, provider in self.providers.items():
                try:
                    articles = await provider.fetch_latest_news(limit=20)
                    all_articles.extend(articles)
                    self.logger.info(f"Collected {len(articles)} articles from {provider_name}")
                except Exception as e:
                    self.logger.error(f"Error collecting news from {provider_name}: {e}")
            
            # Process and deduplicate articles
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Send to callbacks
            for article in unique_articles:
                for callback in self.news_callbacks:
                    try:
                        await callback(article)
                    except Exception as e:
                        self.logger.error(f"Error in news callback: {e}")
            
            # Wait for next collection
            await asyncio.sleep(interval)
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title
            title_key = article.title.lower().strip()[:50]  # First 50 chars
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def add_news_callback(self, callback) -> None:
        """Add callback for processed news"""
        self.news_callbacks.append(callback)
    
    async def stop_news_collection(self) -> None:
        """Stop news collection"""
        self.is_running = False
        self.logger.info("Stopping news collection")
    
    async def close(self) -> None:
        """Close all providers"""
        for provider in self.providers.values():
            await provider.close()
        self.logger.info("News aggregator closed")