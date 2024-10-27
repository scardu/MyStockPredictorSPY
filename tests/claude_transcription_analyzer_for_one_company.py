import yfinance as yf
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta

class EfficientStockAnalyzer:
    def __init__(self):
        # Cargar únicamente el modelo FinBERT para análisis financiero
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True
        )

    def analyze_transcript(self, transcript, stock_symbol=None):
        """
        Analiza una transcripción y retorna análisis de sentimiento y recomendación
        """
        # Dividir el texto en chunks de 512 tokens para procesar textos largos
        chunks = [transcript[i:i+512] for i in range(0, len(transcript), 512)]
        
        # Analizar sentimiento de cada chunk
        sentiments = []
        for chunk in chunks:
            sentiment_scores = self.sentiment_analyzer(chunk)[0]
            sentiments.append({
                score['label']: score['score'] for score in sentiment_scores
            })
        
        # Calcular sentimiento promedio
        avg_sentiment = {
            'positive': sum(s.get('positive', 0) for s in sentiments) / len(sentiments),
            'negative': sum(s.get('negative', 0) for s in sentiments) / len(sentiments),
            'neutral': sum(s.get('neutral', 0) for s in sentiments) / len(sentiments)
        }

        # Obtener datos de mercado si se proporciona símbolo
        market_data = self._get_market_data(stock_symbol) if stock_symbol else None

        return {
            'sentiment': avg_sentiment,
            'recommendation': self._get_recommendation(avg_sentiment),
            'confidence': max(avg_sentiment.values()),
            'market_context': market_data
        }

    def _get_market_data(self, symbol):
        """
        Obtiene datos básicos del mercado
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'current_price': info.get('currentPrice'),
                'target_price': info.get('targetMeanPrice'),
                'sector': info.get('sector'),
                'day_change': info.get('regularMarketChangePercent')
            }
        except:
            return None

    def _get_recommendation(self, sentiment):
        """
        Genera recomendación basada en sentimiento
        """
        pos = sentiment['positive']
        neg = sentiment['negative']
        
        if pos > 0.6:
            return "STRONG_BUY"
        elif pos > 0.4:
            return "BUY"
        elif neg > 0.4:
            return "SELL"
        elif neg > 0.6:
            return "STRONG_SELL"
        else:
            return "HOLD"

    def analyze_multiple(self, transcripts, stock_symbol=None):
        """
        Analiza múltiples transcripciones y retorna un resumen
        """
        analyses = [self.analyze_transcript(t) for t in transcripts]
        
        # Calcular sentimiento promedio
        avg_sentiment = {
            'positive': sum(a['sentiment']['positive'] for a in analyses) / len(analyses),
            'negative': sum(a['sentiment']['negative'] for a in analyses) / len(analyses),
            'neutral': sum(a['sentiment']['neutral'] for a in analyses) / len(analyses)
        }

        return {
            'overall_sentiment': avg_sentiment,
            'recommendation': self._get_recommendation(avg_sentiment),
            'confidence': sum(a['confidence'] for a in analyses) / len(analyses),
            'market_data': self._get_market_data(stock_symbol) if stock_symbol else None
        }
