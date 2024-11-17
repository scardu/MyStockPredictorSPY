from nasdaq_video_analyzer import NasdaqVideoAnalyzer

class Downloader:
    def Download(self):
        analyzer = NasdaqVideoAnalyzer()
        
        # Obtener los resultados de videos analizados
        results_df = analyzer.analyze_recent_nasdaq_content(
            days_back=7,
            min_views=1000
        )
        
        # Guardar los resultados
        analyzer.save_results(results_df)  # Llamada a save_results
        
         # Mostrar resumen
        print("\nResumen de videos analizados:")
        print(f"Total de videos: {len(results_df)}")
        if not results_df.empty:
            print("\nTop 5 videos por views:")
            top_videos = results_df.nlargest(5, 'views')[
                ['title', 'channel', 'views', 'url']
            ]
            print(top_videos.to_string())
            
        # Concatenar las transcripciones de los videos
        glued_transcripts = ""
        for i in range(len(results_df)):
            transcript = results_df["transcript"][i]
            glued_transcripts += transcript
        
        return glued_transcripts
