from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime, timedelta
import pandas as pd
import os
from typing import List, Dict, Any
import json

class NasdaqVideoAnalyzer:
    def __init__(self, api_key: str):
        """
        Inicializar el analizador con la API key de YouTube
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.search_terms = [
            'NASDAQ stock market news',
            'NASDAQ weekly review',
            'NASDAQ stock analysis',
            'NASDAQ market update'
        ]
        
    def search_recent_videos(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Busca videos relevantes de la última semana
        """
        published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + 'Z'
        
        all_videos = []
        for search_term in self.search_terms:
            try:
                request = self.youtube.search().list(
                    q=search_term,
                    type='video',
                    part='id,snippet',
                    maxResults=10,
                    publishedAfter=published_after,
                    relevanceLanguage='en',
                    order='relevance'
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    video_data = {
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'description': item['snippet']['description']
                    }
                    
                    # Obtener estadísticas adicionales del video
                    stats = self.youtube.videos().list(
                        part='statistics',
                        id=item['id']['videoId']
                    ).execute()
                    
                    if stats['items']:
                        video_data.update({
                            'views': stats['items'][0]['statistics'].get('viewCount'),
                            'likes': stats['items'][0]['statistics'].get('likeCount')
                        })
                    
                    all_videos.append(video_data)
                    
            except Exception as e:
                print(f"Error searching for term '{search_term}': {str(e)}")
                continue
                
        # Eliminar duplicados basados en video_id
        unique_videos = {v['video_id']: v for v in all_videos}.values()
        return sorted(unique_videos, key=lambda x: int(x.get('views', 0)), reverse=True)

    def get_transcript(self, video_id: str) -> str:
        """
        Obtiene la transcripción de un video
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en']
            )
            return ' '.join(item['text'] for item in transcript_list)
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {str(e)}")
            return None

    def analyze_recent_nasdaq_content(self, days_back: int = 7, min_views: int = 1000) -> pd.DataFrame:
        """
        Analiza contenido reciente sobre NASDAQ y retorna un DataFrame con los resultados
        """
        # Buscar videos recientes
        videos = self.search_recent_videos(days_back)
        
        # Filtrar por número mínimo de views
        videos = [v for v in videos if int(v.get('views', 0)) >= min_views]
        
        results = []
        for video in videos:
            transcript = self.get_transcript(video['video_id'])
            if transcript:
                video['transcript'] = transcript
                video['url'] = f"https://www.youtube.com/watch?v={video['video_id']}"
                results.append(video)
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        
        # Convertir fechas y números
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['views'] = pd.to_numeric(df['views'])
            df['likes'] = pd.to_numeric(df['likes'])
        
        return df

    def save_results(self, df: pd.DataFrame, output_dir: str = 'nasdaq_analysis'):
        """
        Guarda los resultados en archivos CSV y JSON
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar información básica en CSV
        basic_info = df.drop(columns=['transcript'])
        basic_info.to_csv(f'{output_dir}/videos_info.csv', index=False)
        
        # Guardar transcripciones en archivos JSON separados
        for _, row in df.iterrows():
            video_data = {
                'video_id': row['video_id'],
                'title': row['title'],
                'channel': row['channel'],
                'transcript': row['transcript']
            }
            
            filename = f"{output_dir}/transcript_{row['video_id']}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(video_data, f, ensure_ascii=False, indent=2)

def main():
    # Necesitas obtener una API key de Google Cloud Console
    API_KEY = 'AIzaSyBV6c9-fj7pP7YtFRbZWn-FviZiPvfEfZM'
    
    analyzer = NasdaqVideoAnalyzer(API_KEY)
    
    # Analizar videos de la última semana con al menos 1000 views
    results_df = analyzer.analyze_recent_nasdaq_content(
        days_back=7,
        min_views=1000
    )
    
    # Guardar resultados
    analyzer.save_results(results_df)
    
    # Mostrar resumen
    print("\nResumen de videos analizados:")
    print(f"Total de videos: {len(results_df)}")
    if not results_df.empty:
        print("\nTop 5 videos por views:")
        top_videos = results_df.nlargest(5, 'views')[
            ['title', 'channel', 'views', 'url']
        ]
        print(top_videos.to_string())

if __name__ == "__main__":
    main()