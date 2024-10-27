from claude_transcription_for_multiple_companies import MultiCompanyStockAnalyzer

# Opcional: diccionario de símbolos y nombres de empresas
company_symbols = {
    'AAPL': ['Apple', 'Apple Inc'],
    'MSFT': ['Microsoft', 'Microsoft Corporation'],
    'GOOGL': ['Google', 'Alphabet', 'Alphabet Inc']
}

#analyzer = MultiCompanyStockAnalyzer(company_symbols)

analyzer = MultiCompanyStockAnalyzer(None)


# Analizar una transcripción
transcript = """
En las noticias de hoy, Apple ha anunciado nuevos productos que han impresionado al mercado.
Microsoft continúa su expansión en la nube, mientras que Google enfrenta nuevos desafíos regulatorios.
"""

results = analyzer.analyze_transcript(transcript)

# Mostrar resultados por empresa
for company, analysis in results.items():
    print(f"\nAnálisis para {company}:")
    print(f"Sentimiento: {analysis['sentiment']}")
    print(f"Recomendación: {analysis['recommendation']}")
    print(f"Menciones: {analysis['mention_count']}")
    if analysis['market_data']:
        print(f"Precio actual: ${analysis['market_data']['current_price']}")
print("output")
print(results)