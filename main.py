from tunespotter import TuneSpotter
import sys

if sys.argv[1] == "--dataset":
    ims = TuneSpotter.collect_data()
elif sys.argv[1] == "--detect":
    print("[LOGGER] Generating spectrogram from audio file")
    ims = TuneSpotter.plotstft(sys.argv[2])
    print("[LOGGER] Comparing the audio signature with existing ones")
    TuneSpotter.match_graph()
else:
    print("""
    [ERROR] Arguments weren't given
    
    python main.py --dataset 
    to generate a dataset
    
    python main.py --detect filename.wav
    to identify the song
    """)
