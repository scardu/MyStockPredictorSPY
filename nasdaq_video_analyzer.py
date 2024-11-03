from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime, timedelta
import pandas as pd
import os
from typing import List, Dict, Any, Optional
import json

class NasdaqVideoAnalyzer:
    def __init__(self):
        """
        Inicializar el analizador con la API key de YouTube
        """
        API_KEY = 'AIzaSyBV6c9-fj7pP7YtFRbZWn-FviZiPvfEfZM'
        self.youtube = build('youtube', 'v3', developerKey=API_KEY)
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
                    try:
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
                            statistics = stats['items'][0]['statistics']
                            video_data.update({
                                'views': statistics.get('viewCount', '0'),
                                'likes': statistics.get('likeCount', '0')
                            })
                        else:
                            video_data.update({
                                'views': '0',
                                'likes': '0'
                            })
                        
                        all_videos.append(video_data)
                        
                    except Exception as e:
                        print(f"Error procesando video individual: {str(e)}")
                        continue
                    
            except Exception as e:
                print(f"Error buscando término '{search_term}': {str(e)}")
                continue
                
        # Eliminar duplicados basados en video_id y manejar valores nulos
        unique_videos = {v['video_id']: v for v in all_videos}.values()
        
        # Asegurar que 'views' sea un string válido antes de convertir a int
        return sorted(
            unique_videos,
            key=lambda x: int(x.get('views', '0')),
            reverse=True
        )

    def get_transcript(self, video_id: str) -> Optional[str]:
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
            print(f"Error obteniendo transcripción del video {video_id}: {str(e)}")
            return None

    def analyze_recent_nasdaq_content(self, days_back: int = 7, min_views: int = 1000) -> pd.DataFrame:
        """
        Analiza contenido reciente sobre NASDAQ y retorna un DataFrame con los resultados
        """
        try:
            # Buscar videos recientes
            videos = self.search_recent_videos(days_back)
            
            # Filtrar por número mínimo de views
            videos = [v for v in videos if int(v.get('views', '0')) >= min_views]
            
            # Seleccionar solo los primeros 3 videos
            videos = videos[:10]
            
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
            
        except Exception as e:
            print(f"Error en analyze_recent_nasdaq_content: {str(e)}")
            return pd.DataFrame()  # Retornar DataFrame vacío en caso de error


    def save_results(self, df: pd.DataFrame, output_dir: str = 'nasdaq_analysis'):
        """
        Guarda los resultados en archivos CSV y JSON
        """
        try:
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
                    
        except Exception as e:
            print(f"Error guardando resultados: {str(e)}")

def main():
    try:
        analyzer = NasdaqVideoAnalyzer()
        
        # Analizar videos de la última semana con al menos 1000 views
        results_df = analyzer.analyze_recent_nasdaq_content(
            days_back=5,
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
            
    except Exception as e:
        print(f"Error en main: {str(e)}")

if __name__ == "__main__":
    main()
