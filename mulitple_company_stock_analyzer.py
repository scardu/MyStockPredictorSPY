import yfinance as yf
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import spacy
import re
from concurrent.futures import ThreadPoolExecutor  # Importar ThreadPoolExecutor
import requests
import time
import numpy as np



API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": "Bearer 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"}

class MultiCompanyStockAnalyzer:
    def __init__(self, company_symbols=None, chunk_size=900000):
        # Cargar el modelo FinBERT para análisis financiero
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True
        )

        #self.sentiment_analyzer = self.__remote_sentiment_analyzer
        
        # Cargar modelo spaCy para NER (Named Entity Recognition)
        self.nlp = spacy.load("en_core_web_sm")
        self.chunk_size = chunk_size
        
        # Lista de símbolos de empresas conocidas (opcional)
        self.company_symbols = {
                    'AAPL': ['apple', 'apple inc.'],
                    'MSFT': ['microsoft', 'microsoft corporation'],
                    'AMZN': ['amazon', 'amazon.com'],
                    'GOOG': ['google', 'alphabet'],
                    'TSLA': ['tesla'],
                    'NVDA': ['nvidia'],
                    'PFE': ['pfizer'],
                    'JNJ': ['johnson & johnson'],
                    'UNH': ['unitedhealth group'],
                    'PYPL': ['paypal'],
                    'COST': ['costco'],
                    'HD': ['home depot'],
                    'DIS': ['disney'],
                    'NFLX': ['netflix'],
                    'ADBE': ['adobe'],
                    'CMCSA': ['comcast'],
                    'PEP': ['pepsico'],
                    'MCD': ['mcdonald''s'],
                    'INTC': ['intel'],
                    'CSCO': ['cisco'],
                    'ABBV': ['abbvie'],
                    'AVGO': ['broadcom'],
                    'TMO': ['thermo fisher scientific'],
                    'QCOM': ['qualcomm'],
                    'AMGN': ['amgen'],
                    'SHOP': ['shopify'],
                    'MRNA': ['moderna'],
                    'TMUS': ['t-mobile'],
                    'ISRG': ['intuitive surgical'],
                    'AMD': ['advanced micro devices'],
                    'MDLZ': ['mondelez international'],
                    'SBUX': ['starbucks'],
                    'BKNG': ['booking holdings', 'booking'],
                    'TXN': ['texas instruments'],
                    'ATVI': ['activision blizzard'],
                    'GILD': ['gilead sciences'],
                    'VRTX': ['vertex pharmaceuticals'],
                    'REGN': ['regeneron pharmaceuticals'],
                    'EBAY': ['ebay'],
                    'CHTR': ['charter communications'],
                    'LRCX': ['lam research'],
                    'ADSK': ['autodesk'],
                    'SNAP': ['snap'],
                    'CTSH': ['cognizant'],
                    'LULU': ['lululemon athletica'],
                    'KHC': ['kraft heinz'],
                    'DXCM': ['dexcom'],
                    'FOXA': ['fox'],
                    'ZM': ['zoom video communications', 'Zoom'],
                    'NXPI': ['nxp semiconductors'],
                    'AMAT': ['applied materials'],
                    'FAST': ['fastenal'],
                    'FISV': ['fiserv'],
                    'CPRT': ['copart'],
                    'ASML': ['asml holding'],
                    'CRWD': ['crowdstrike'],
                    'SGEN': ['seagen'],
                    'MELI': ['mercadolibre'],
                    'PANW': ['palo alto networks'],
                    'PATH': ['uipath'],
                    'FTNT': ['fortinet'],
                    'DDOG': ['datadog'],
                    'KLAC': ['kla'],
                    'INTU': ['intuit'],
                    'DOCU': ['docusign'],
                    'SPLK': ['splunk'],
                    'CTAS': ['cintas'],
                    'TEAM': ['atlassian'],
                    'NDAQ': ['nasdaq'],
                    'ZS': ['zscaler'],
                    'FIVN': ['five9'],
                    'ETSY': ['etsy'],
                    'PAYC': ['paycom software'],
                    'CDNS': ['cadence design systems'],
                    'BIIB': ['biogen'],
                    'PTON': ['peloton'],
                    'XPEV': ['xpeng'],
                    'OKTA': ['okta'],
                    'SMAR': ['smartsheet'],
                    'ABNB': ['airbnb'],
                    'VEEV': ['veeva systems'],
                    'PINS': ['pinterest'],
                    'PLTR': ['palantir technologies'],
                    'UBER': ['uber'],
                    'COIN': ['coinbase global'],
                    'RIVN': ['rivian automotive'],
                    'TWLO': ['twilio'],
                    'CRSP': ['crispr therapeutics'],
                    'CVNA': ['carvana'],
                    'RBLX': ['roblox'],
                    'SOFI': ['sofi technologies'],
                    'DKNG': ['draftkings'],
                    'FVRR': ['fiverr international'],
                    'CRWD': ['crowdstrike holdings'],
                    'MTCH': ['match group'],
                    'ROKU': ['roku'],
                    'UPST': ['upstart'],
                    'TTWO': ['take-two interactive software'],
                    'MRNA': ['moderna'],
                    'DASH': ['doordash'],
                    'DXCM': ['dexcom'],
                    'EXPE': ['expedia group'],
                    'BKNG': ['booking holdings'],
                    'WBD': ['warner bros. discovery'],
                    'HOOD': ['robinhood markets'],
                    'ADBE': ['adobe'],
                    'STNE': ['stoneco'],
                    'ABNB': ['airbnb'],
                    'HOOD': ['robinhood markets'],
                    'ZY': ['zynga'],
                    'CHWY': ['chewy'],
                    'COUP': ['coupa software'],
                    'AFRM': ['affirm holdings'],
                    'SAVA': ['cassava sciences'],
                    'DUOL': ['duolingo'],
                    'AMWL': ['amwell'],
                    'FRPT': ['freshpet'],
                    'LCID': ['lucid group'],
                    'RBLX': ['roblox'],
                    'GDRX': ['goodrx holdings'],
                    'XPEV': ['xpeng'],
                    'LMND': ['lemonade'],
                    'WEBR': ['weber'],
                    'WISH': ['contextlogic'],
                    'PTRA': ['proterra'],
                    'BOWX': ['boxwood acquisition corp'],
                    'IONQ': ['ionq'],
                    'CLOV': ['clover health investments'],
                    'DWAC': ['digital world acquisition corp'],
                    'SKLZ': ['skillz'],
                    'ASTS': ['ast spacemobile'],
                    'FIGS': ['figs'],
                    'UPST': ['upstart'],
                    'IIPR': ['innovative industrial properties'],
                    'STLD': ['steel dynamics'],
                    'RXT': ['rackspace technology'],
                    'TLS': ['telos'],
                    'NEWR': ['new relic'],
                    'GTLB': ['gitlab'],
                    'ARCT': ['arcturus therapeutics'],
                    'SLQT': ['selectquote'],
                    'ASTS': ['ast spacemobile'],
                    'EVBG': ['everbridge']
                }
        
        # Crear diccionario inverso de nombres a símbolos
        self.company_names = {
            name.lower(): symbol 
            for symbol, names in self.company_symbols.items() 
            for name in names
        }

    def __query_remote_finbert(self, payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def __remote_sentiment_analyzer(self, text):
        output = self.__query_remote_finbert({
            "inputs": text
        })
        print("chunk processed")
        return output
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
        
        return [c for c in companies if c in self.company_symbols]

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
        total_start_time = time.time()
        metrics = {}

        # Extraer contextos por empresa
        context_start = time.time()
        company_contexts = self.extract_company_contexts(transcript)
        context_time = time.time() - context_start
        metrics['context_extraction'] = context_time
        print(f"1. Tiempo para extraer contextos: {context_time:.2f} segundos")
        print(f"   Número de empresas encontradas: {len(company_contexts)}")
        results = {}
        total_contexts = 0
        total_chunks = 0

        for company, contexts in company_contexts.items():
            # Analizar cada contexto de la empresa en paralelo
            if company in self.company_symbols:
                print(f"\nProcesando empresa: {company}")
                company_start = time.time()
                total_contexts += len(contexts)
                sentiments = []
                chunk_times = []
                chunk_start = time.time()
            # Usar ThreadPoolExecutor para paralelizar el análisis de sentimientos
            with ThreadPoolExecutor() as executor:
                # Dividir en chunks de 512 caracteres y analizar en paralelo
                for context_index, context in enumerate(contexts):
                    print(f"   Analizando contexto {context_index + 1}/{len(contexts)} para {company}...")
                    chunks = [context[i:i+512] for i in range(0, len(context), 512)]
                    total_chunks += len(chunks)
                    batch_start = time.time()
                    future_results = list(executor.map(self.sentiment_analyzer, chunks))
                    chunk_time = time.time() - batch_start
                    chunk_times.append(chunk_time)
                    for chunk_index, chunk in enumerate(chunks):
                        try:
                            chunk_start = time.time()
                            sentiment_scores = self.sentiment_analyzer(chunk)
                            chunk_time = time.time() - chunk_start
                            chunk_times.append(chunk_time)
                            print(f"      Chunk {chunk_index + 1}/{len(chunks)} procesado en {chunk_time:.2f} segundos.")
                            if sentiment_scores:
                                sentiments.append({
                                score['label']: score['score'] 
                                for score in sentiment_scores[0]
                            })
                        except Exception as e:
                            print(f"      Error procesando chunk {chunk_index + 1}: {e}")
            chunk_processing_time = time.time() - chunk_start
            # Calcular sentimiento promedio para la empresa
            if sentiments:  # Verificar que hay sentimientos para analizar
                avg_sentiment = {
                    'positive': sum(s.get('positive', 0) for s in sentiments) / len(sentiments),
                    'negative': sum(s.get('negative', 0) for s in sentiments) / len(sentiments),
                    'neutral': sum(s.get('neutral', 0) for s in sentiments) / len(sentiments)
                }

                results[company] = {
                    'sentiment': avg_sentiment,
                    'recommendation': self._get_recommendation(avg_sentiment),
                    'confidence': max(avg_sentiment.values()),
                    'mention_count': len(contexts)
                }
                print(f"   Sentimiento promedio para {company}: {avg_sentiment}")
            company_time = time.time() - company_start
            print(f"2. Procesamiento de empresa {company}:")
            print(f"   - Número de contextos: {len(contexts)}")
            print(f"   - Número de chunks: {len(chunks)}")
            print(f"   - Tiempo promedio por chunk: {np.mean(chunk_times):.2f} segundos")
            print(f"   - Tiempo total de la empresa: {company_time:.2f} segundos")

        total_time = time.time() - total_start_time
        print("\nResumen de rendimiento:")
        print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
        print(f"Total de empresas procesadas: {len(company_contexts)}")
        print(f"Total de contextos analizados: {total_contexts}")
        print(f"Total de chunks procesados: {total_chunks}")
        print(f"Tiempo promedio por chunk: {(total_time/total_chunks if total_chunks else 0):.2f} segundos")

        results['_metrics'] = {
          'total_time': total_time,
          'context_extraction_time': context_time,
          'companies_processed': len(company_contexts),
          'total_contexts': total_contexts,
          'total_chunks': total_chunks,
          'avg_time_per_chunk': total_time/total_chunks if total_chunks else 0
        }



      
        return results

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
