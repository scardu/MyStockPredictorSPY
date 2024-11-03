from nasdaq_video_analyzer import NasdaqVideoAnalyzer

class Downloader:
    def Download(self):
        analyzer = NasdaqVideoAnalyzer()
        results_df = analyzer.analyze_recent_nasdaq_content(
        days_back=7,
        min_views=1000
    )
        glued_transcripts = ""
        for i in range(len(results_df)):
            transcript = results_df["transcript"][i]
            glued_transcripts += transcript
        return glued_transcripts