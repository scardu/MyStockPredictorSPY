from tests.claude_transcription_analyzer_for_one_company import EfficientStockAnalyzer

# Crear el analizador
analyzer = EfficientStockAnalyzer()

# Analizar una transcripción
transcript = "Strong quarterly results with revenue up 25%. Our new product line..."
result = analyzer.analyze_transcript(transcript, "AAPL")
print(f"Recomendación: {result['recommendation']}")
print(f"Confianza: {result['confidence']:.2%}")
print(result)

# Analizar múltiples transcripciones
transcripts = [
    "First earnings call transcript...",
    "Second analyst review..."
]
full_analysis = analyzer.analyze_multiple(transcripts, "AAPL")