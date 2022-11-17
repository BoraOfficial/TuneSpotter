import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import cv2
import os
import warnings
import time



warnings.simplefilter("ignore")

class TuneSpotter():
    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize)) 

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
        samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
        # cols for windowing
        cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))   

        frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
        frames *= win   

        return np.fft.rfft(frames)      

    """ scale frequency axis logarithmically """    
    def logscale_spec(spec, sr=44100, factor=20.):
        timebins, freqbins = np.shape(spec) 

        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins-1)/max(scale)
        scale = np.unique(np.round(scale))  

        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):        
            if i == len(scale)-1:
                newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
            else:        
                newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)    

        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])] 

        return newspec, freqs   

    """ plot spectrogram"""
    def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
        samplerate, samples = wav.read(audiopath)   

        s = TuneSpotter.stft(samples, binsize)  

        sshow, freq = TuneSpotter.logscale_spec(s, factor=1.0, sr=samplerate)   

        ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel  

        timebins, freqbins = np.shape(ims)  

        #print("timebins: ", timebins)
        #print("freqbins: ", freqbins)  
    

        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        plt.colorbar()  

        #plt.xlabel("time (s)")
        #plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins]) 

        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
        ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])   

        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])    
    
    

        plt.savefig('__cache__.png', bbox_inches='tight', pad_inches=0)
        #plt.show() 

        plt.clf()   

        return ims  
    # another function to plot spectrogram
    def genplotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
        samplerate, samples = wav.read(audiopath)   

        s = TuneSpotter.stft(samples, binsize)  

        sshow, freq = TuneSpotter.logscale_spec(s, factor=1.0, sr=samplerate)   

        ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel  

        timebins, freqbins = np.shape(ims)  

        #print("timebins: ", timebins)
        #print("freqbins: ", freqbins)  
    

        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        plt.colorbar()  

        #plt.xlabel("time (s)")
        #plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins]) 

        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
        ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])   

        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])    
    
    

        plt.savefig('dataset\\'+audiopath.replace("songs\\", "").split(".")[0].capitalize()+'.png', bbox_inches='tight', pad_inches=0)
        #plt.show() 

        plt.clf()   

        return ims
    def match_graph():
        for filename in os.listdir("dataset\\"):
            if filename.endswith(".png"): 
                a = cv2.imread(os.path.abspath("dataset\\"+filename))
                b = cv2.imread(os.path.abspath("__cache__.png"))
                difference = cv2.subtract(a, b)    
                result = not np.any(difference)
                if result is True:
                    os.system('cls' if os.name=='nt' else 'clear')
                    print(f'Determined the song as "{filename.split(".")[0]}"')
                    exit()  

                continue
            else:
                continue    

        print("[LOGGER] Audio signature wasn't found in the dataset")



    def folder_size(path='.'):
        total = 0
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += folder_size(entry.path)
        return total    

    def collect_data():
        os.system('cls' if os.name=='nt' else 'clear')
        print("-"*100)
        if os.path.isdir('dataset') != True:
            os.mkdir("dataset")
            print(f'[LOGGER] Created folder "dataset" ({time.strftime("%Y/%m/%d %H:%M")})')
        else:
            print(f'[LOGGER] Folder "dataset" exists ({time.strftime("%Y/%m/%d %H:%M")})')  

        if os.path.isdir('songs') != True:
            
            raise ExceptionHandler("Couldn't find the songs, check if they are located inside the 'songs' folder")
        else:
            print(f'[LOGGER] Loaded songs from "songs" ({time.strftime("%Y/%m/%d %H:%M")})')    
    
    
    

        print(f'[LOGGER] Started creating datasets from songs ({time.strftime("%Y/%m/%d %H:%M")})')
        print("-"*100+"\n")
        for filename in os.listdir("songs\\"):
            if filename.endswith(".wav"): 
                start_time = time.time()
                TuneSpotter.genplotstft("songs\\"+filename)
                print(f'[LOGGER] Completed data creation from song "{filename.split(".")[0]}" in {round(time.time() - start_time, 2)} seconds ({time.strftime("%Y/%m/%d %H:%M")})')
                print("\n"+"-"*100) 

                continue
            else:
                continue

