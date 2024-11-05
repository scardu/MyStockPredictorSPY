from predictor import Predictor

predictor = Predictor()

transcript = predictor.DownloadVideos()
analysis = predictor.Analyze(transcript)
predictor.ShowPredictions(analysis)

