import yfinance as yf
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import spacy
import re

class MultiCompanyStockAnalyzer:
    def __init__(self, company_symbols=None, chunk_size=900000):
        # Cargar el modelo FinBERT para análisis financiero
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True
        )
        
        # Cargar modelo spaCy para NER (Named Entity Recognition)
        self.nlp = spacy.load("en_core_web_sm")
        self.chunk_size = chunk_size
        
        # Lista de símbolos de empresas conocidas (opcional)
        self.company_symbols = company_symbols or {}
        
        # Crear diccionario inverso de nombres a símbolos
        self.company_names = {
            name.lower(): symbol 
            for symbol, names in self.company_symbols.items() 
            for name in names
        }

    def chunk_text(self, text):
        """
        Divide el texto en fragmentos más pequeños, intentando mantener oraciones completas
        """
        chunks = []
        current_pos = 0
        text_length = len(text)

        while current_pos < text_length:
            # Encuentra el último punto antes del límite del chunk
            chunk_end = min(current_pos + self.chunk_size, text_length)
            if chunk_end < text_length:
                # Buscar el último punto en los últimos 100 caracteres del chunk
                last_period = text.rfind('.', current_pos, chunk_end)
                if last_period != -1:
                    chunk_end = last_period + 1

            chunks.append(text[current_pos:chunk_end])
            current_pos = chunk_end

        return chunks

    def identify_companies(self, text):
        """
        Identifica menciones de empresas en el texto usando NER y matching con símbolos conocidos
        """
        companies = set()
        
        # Dividir el texto en chunks manejables
        chunks = self.chunk_text(text)
        
        for chunk in chunks:
            # Procesar cada chunk con spaCy
            doc = self.nlp(chunk)
            
            # Extraer organizaciones identificadas por spaCy
            for ent in doc.ents:
                if ent.label_ in ["ORG"]:
                    company_name = ent.text.lower()
                    # Si tenemos el símbolo para esta empresa, usarlo
                    if company_name in self.company_names:
                        companies.add(self.company_names[company_name])
                    else:
                        companies.add(ent.text)
            
            # Buscar símbolos de bolsa directamente (formato: $AAPL o AAPL)
            stock_symbols = re.findall(r'[\$]?([A-Z]{1,5})\b', chunk)
            companies.update(stock_symbols)
        
        return list(companies)

    def extract_company_contexts(self, text, window_size=1000):
        """
        Extrae el contexto alrededor de cada mención de empresa
        """
        company_contexts = {}
        
        # Primero identificar todas las menciones de empresas
        companies = self.identify_companies(text)
        
        # Dividir el texto en chunks para el procesamiento
        chunks = self.chunk_text(text)
        full_context = ""  # Para mantener un contexto continuo entre chunks
        
        for chunk in chunks:
            full_context += chunk
            
            for company in companies:
                # Buscar menciones de la empresa (incluyendo variaciones)
                patterns = [company]
                if company in self.company_symbols:
                    patterns.extend(self.company_symbols[company])
                
                if company not in company_contexts:
                    company_contexts[company] = []
                
                for pattern in patterns:
                    # Encontrar todas las menciones en el chunk actual
                    for match in re.finditer(re.escape(pattern), chunk, re.IGNORECASE):
                        start = max(0, match.start() - window_size)
                        end = min(len(chunk), match.end() + window_size)
                        context = chunk[start:end]
                        company_contexts[company].append(context)
        
        return company_contexts

    def analyze_transcript(self, transcript):
        """
        Analiza una transcripción y retorna análisis por empresa
        """
        # Extraer contextos por empresa
        company_contexts = self.extract_company_contexts(transcript)
        
        results = {}
        for company, contexts in company_contexts.items():
            # Analizar cada contexto de la empresa
            sentiments = []
            for context in contexts:
                # Dividir en chunks si es necesario (para FinBERT que tiene límite de 512 tokens)
                chunks = [context[i:i+512] for i in range(0, len(context), 512)]
                
                for chunk in chunks:
                    sentiment_scores = self.sentiment_analyzer(chunk)[0]
                    sentiments.append({
                        score['label']: score['score'] for score in sentiment_scores
                    })
            
            # Calcular sentimiento promedio para la empresa
            if sentiments:  # Verificar que hay sentimientos para analizar
                avg_sentiment = {
                    'positive': sum(s.get('positive', 0) for s in sentiments) / len(sentiments),
                    'negative': sum(s.get('negative', 0) for s in sentiments) / len(sentiments),
                    'neutral': sum(s.get('neutral', 0) for s in sentiments) / len(sentiments)
                }
                
                # Obtener datos de mercado
               ## market_data = self._get_market_data(company)
                
                results[company] = {
                    'sentiment': avg_sentiment,
                    'recommendation': self._get_recommendation(avg_sentiment),
                    'confidence': max(avg_sentiment.values()),
                    ##'market_data': market_data,
                    'mention_count': len(contexts)
                }
        
        return results

# def _get_market_data(self, symbol):
#     """
#     Obtiene datos básicos del mercado
#     """
#     try:
#         stock = yf.Ticker(symbol)
#         info = stock.info
#         return {
#             'current_price': info.get('currentPrice'),
#             'target_price': info.get('targetMeanPrice'),
#             'sector': info.get('sector'),
#             'day_change': info.get('regularMarketChangePercent')
#         }
#     except:
#         return None

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