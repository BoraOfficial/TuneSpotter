# TuneSpotter
---

Shazam-like music identifier written in <a href="https://python.org">Python</a>.

---
## How to use?
---
First, download the repository and extract the files, then open up a terminal inside the repository's folder. When that's done, you need to create your own song dataset. To do that, you need to first create a folder called <b>"songs",</b> then move every song you have downloaded in it. They all need to be <b>.wav</b> files.

After that, just run the code and it'll do the rest for you.
To generate a dataset use this command:

```
python main.py --dataset
```

After the dataset is generated, you can now identify any song as long as it's in the dataset.
To do that just use this command:

```
python main.py --detect enter-file-name-here.wav
```

---
This code is only in the prototype stage. Currently, you need to upload a file with the exact notes. 
