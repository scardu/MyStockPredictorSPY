import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"


from predictor import Predictor

predictor = Predictor()

transcript = predictor.DownloadVideos()
analysis = predictor.Analyze(transcript)
predictor.ShowPredictions(analysis)

