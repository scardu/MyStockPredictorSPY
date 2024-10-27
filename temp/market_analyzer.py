import yfinance as yf
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from transformers import pipeline
import spacy
import re
from typing import List, Dict, Any
import json
from coinbase.wallet.client import Client
import requests
from concurrent.futures import ThreadPoolExecutor

class MarketAnalyzer:
    def __init__(self, youtube_api_key: str, coinbase_api_key: str = None, coinbase_secret: str = None):
        """
        Inicializar el analizador con las API keys necesarias
        """
        # APIs y modelos
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True
        )
        
        # Configuración Coinbase (opcional)
        if coinbase_api_key and coinbase_secret:
            self.coinbase_client = Client(coinbase_api_key, coinbase_secret)
        else:
            self.coinbase_client = None
            
        # Términos de búsqueda para YouTube
        self.stock_search_terms = [
            'NASDAQ stock market analysis',
            'NASDAQ trading opportunities',
            'stock market predictions this week'
        ]
        
        self.crypto_search_terms = [
            'cryptocurrency market analysis',
            'crypto trading opportunities',
            'bitcoin ethereum analysis'
        ]
        
        # Listas de símbolos importantes
        self.major_stocks = self._get_nasdaq_top_stocks()
        self.major_cryptos = self._get_major_cryptos()

    def _get_nasdaq_top_stocks(self) -> List[str]:
        """
        Obtiene lista de principales acciones del NASDAQ
        """
        try:
            # Obtener los componentes principales del NASDAQ-100
            nasdaq100 = pd.read_html(
                'https://en.wikipedia.org/wiki/Nasdaq-100'
            )[4]['Ticker']
            return nasdaq100.tolist()
        except:
            # Lista de respaldo con principales empresas
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    def _get_major_cryptos(self) -> List[str]:
        """
        Obtiene lista de principales criptomonedas
        """
        try:
            # Obtener top criptos de CoinGecko
            url = 'https://api.coingecko.com/api/v3/coins/markets'
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 20,
                'page': 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            return [coin['symbol'].upper() for coin in data]
        except:
            # Lista de respaldo
            return ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL']

    def get_recent_videos(self, search_terms: List[str], days_back: int = 7) -> List[Dict]:
        """
        Obtiene videos recientes basados en términos de búsqueda
        """
        published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + 'Z'
        all_videos = []
        
        for term in search_terms:
            try:
                request = self.youtube.search().list(
                    q=term,
                    type='video',
                    part='id,snippet',
                    maxResults=10,
                    publishedAfter=published_after,
                    relevanceLanguage='en',
                    order='relevance'
                )
                response = request.execute()
                
                for item in response['items']:
                    video_data = {
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt']
                    }
                    
                    # Obtener transcripción
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(
                            video_data['video_id'], 
                            languages=['en']
                        )
                        video_data['transcript'] = ' '.join(
                            item['text'] for item in transcript_list
                        )
                        all_videos.append(video_data)
                    except:
                        continue
                        
            except Exception as e:
                print(f"Error en búsqueda de '{term}': {str(e)}")
                continue
                
        return all_videos

    def analyze_sentiment(self, text: str, symbols: List[str]) -> Dict:
        """
        Analiza sentimiento para símbolos específicos en el texto
        """
        results = {}
        
        for symbol in symbols:
            # Buscar menciones del símbolo y contexto cercano
            mentions = re.finditer(rf'\b{symbol}\b', text, re.IGNORECASE)
            contexts = []
            
            for match in mentions:
                start = max(0, match.start() - 500)
                end = min(len(text), match.end() + 500)
                contexts.append(text[start:end])
            
            if contexts:
                # Analizar sentimiento para cada contexto
                sentiments = []
                for context in contexts:
                    chunks = [context[i:i+512] for i in range(0, len(context), 512)]
                    for chunk in chunks:
                        sentiment_scores = self.sentiment_analyzer(chunk)[0]
                        sentiments.append({
                            score['label']: score['score'] 
                            for score in sentiment_scores
                        })
                
                # Calcular promedio
                avg_sentiment = {
                    'positive': np.mean([s['positive'] for s in sentiments]),
                    'negative': np.mean([s['negative'] for s in sentiments]),
                    'neutral': np.mean([s['neutral'] for s in sentiments])
                }
                
                results[symbol] = {
                    'sentiment': avg_sentiment,
                    'mention_count': len(contexts),
                    'sentiment_strength': max(
                        avg_sentiment['positive'],
                        avg_sentiment['negative']
                    )
                }
        
        return results

    def get_market_data(self, symbol: str, is_crypto: bool = False) -> Dict:
        """
        Obtiene datos de mercado para un símbolo
        """
        try:
            if is_crypto:
                # Usar API de CoinGecko para criptos
                url = f'https://api.coingecko.com/api/v3/simple/price'
                params = {
                    'ids': symbol.lower(),
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true'
                }
                response = requests.get(url, params=params)
                data = response.json()
                
                return {
                    'current_price': data[symbol.lower()]['usd'],
                    'change_24h': data[symbol.lower()]['usd_24h_change']
                }
            else:
                # Usar yfinance para acciones
                stock = yf.Ticker(symbol)
                info = stock.info
                return {
                    'current_price': info.get('currentPrice'),
                    'target_price': info.get('targetMeanPrice'),
                    'recommendation': info.get('recommendationKey'),
                    'change_24h': info.get('regularMarketChangePercent')
                }
        except:
            return None

    def generate_recommendations(self, sentiment_results: Dict, 
                              market_data: Dict, threshold: float = 0.6) -> Dict:
        """
        Genera recomendaciones basadas en sentimiento y datos de mercado
        """
        recommendations = {
            'strong_buy': [],
            'buy': [],
            'hold': [],
            'sell': [],
            'strong_sell': []
        }
        
        for symbol, data in sentiment_results.items():
            sentiment = data['sentiment']
            market = market_data.get(symbol, {})
            
            # Calcular score combinado
            sentiment_score = sentiment['positive'] - sentiment['negative']
            market_score = 0
            
            if market:
                # Normalizar cambio de precio a un rango de -1 a 1
                price_change = market.get('change_24h', 0)
                if price_change:
                    market_score = max(min(price_change / 10, 1), -1)
            
            combined_score = (sentiment_score + market_score) / 2
            
            # Clasificar basado en score combinado
            if combined_score > threshold:
                recommendations['strong_buy'].append(symbol)
            elif combined_score > threshold/2:
                recommendations['buy'].append(symbol)
            elif combined_score < -threshold:
                recommendations['strong_sell'].append(symbol)
            elif combined_score < -threshold/2:
                recommendations['sell'].append(symbol)
            else:
                recommendations['hold'].append(symbol)
        
        return recommendations

    def analyze_market(self, market_type: str = 'stocks') -> Dict:
        """
        Analiza el mercado completo (stocks o crypto)
        """
        # Configurar parámetros según el tipo de mercado
        if market_type == 'stocks':
            search_terms = self.stock_search_terms
            symbols = self.major_stocks
            is_crypto = False
        else:
            search_terms = self.crypto_search_terms
            symbols = self.major_cryptos
            is_crypto = True
        
        # Obtener y analizar videos recientes
        videos = self.get_recent_videos(search_terms)
        all_text = ' '.join(video['transcript'] for video in videos)
        
        # Analizar sentimiento
        sentiment_results = self.analyze_sentiment(all_text, symbols)
        
        # Obtener datos de mercado
        market_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.get_market_data, symbol, is_crypto): symbol 
                for symbol in symbols
            }
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    market_data[symbol] = future.result()
                except:
                    continue
        
        # Generar recomendaciones
        recommendations = self.generate_recommendations(
            sentiment_results,
            market_data
        )
        
        return {
            'recommendations': recommendations,
            'sentiment_analysis': sentiment_results,
            'market_data': market_data,
            'analysis_date': datetime.now().isoformat(),
            'videos_analyzed': len(videos)
        }

def main():
    # Configurar API keys
    YOUTUBE_API_KEY = 'AIzaSyBV6c9-fj7pP7YtFRbZWn-FviZiPvfEfZM'
    COINBASE_API_KEY = 'TU_API_KEY_COINBASE'  # Opcional
    COINBASE_SECRET = 'TU_SECRET_COINBASE'    # Opcional
    
    # Inicializar analizador
    analyzer = MarketAnalyzer(
        YOUTUBE_API_KEY,
        COINBASE_API_KEY,
        COINBASE_SECRET
    )
    
    # Analizar stocks
    print("Analizando mercado de acciones...")
    stock_analysis = analyzer.analyze_market('stocks')
    
    print("\nRecomendaciones para acciones:")
    for category, symbols in stock_analysis['recommendations'].items():
        if symbols:
            print(f"\n{category.upper()}:")
            for symbol in symbols:
                sentiment = stock_analysis['sentiment_analysis'][symbol]
                market_data = stock_analysis['market_data'].get(symbol, {})
                print(f"- {symbol}")
                print(f"  Sentimiento positivo: {sentiment['sentiment']['positive']:.2f}")
                print(f"  Menciones: {sentiment['mention_count']}")
                if market_data:
                    print(f"  Precio actual: ${market_data.get('current_price', 'N/A')}")
                    print(f"  Cambio 24h: {market_data.get('change_24h', 'N/A')}%")
    
    # Analizar criptomonedas
    print("\nAnalizando mercado de criptomonedas...")
    crypto_analysis = analyzer.analyze_market('crypto')
    
    print("\nRecomendaciones para criptomonedas:")
    for category, symbols in crypto_analysis['recommendations'].items():
        if symbols:
            print(f"\n{category.upper()}:")
            for symbol in symbols:
                sentiment = crypto_analysis['sentiment_analysis'][symbol]
                market_data = crypto_analysis['market_data'].get(symbol, {})
                print(f"- {symbol}")
                print(f"  Sentimiento positivo: {sentiment['sentiment']['positive']:.2f}")
                print(f"  Menciones: {sentiment['mention_count']}")
                if market_data:
                    print(f"  Precio actual: ${market_data.get('current_price', 'N/A')}")
                    print(f"  Cambio 24h: {market_data.get('change_24h', 'N/A')}%")
    
    # Guardar resultados
    results = {
        'stocks': stock_analysis,
        'crypto': crypto_analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('market_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()