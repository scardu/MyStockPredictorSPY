from analyzer import Analyzer
from downloader import Downloader

class Predictor:
    def __init__(self):
        self.analyzer = Analyzer()

    def DownloadVideos(self):
        self.downloader = Downloader()
        print("Downloading videos")
        result = self.downloader.Download()
        return result

    def Analyze(self, transcript):
        return self.analyzer.Analyze(transcript)

    def ShowPredictions(self, analysis):
        print("Showing predictions")
        print(analysis)
    