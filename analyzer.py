from mulitple_company_stock_analyzer import MultiCompanyStockAnalyzer

class Analyzer:
    def __init__(self):
        self.multiple_company_analyzer = MultiCompanyStockAnalyzer()

    def Analyze(self, transcript):
        print("Analyzing videos")
        return self.multiple_company_analyzer.analyze_transcript(transcript)