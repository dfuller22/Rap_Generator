# Rap Generator
RNN for generating rap lyrics by Darius Fuller
 
## Background
I am someone who *loves* to music. One of my favorite genres to listen to is hip-hop/rap because of, in most cases, the focus on lyricism. 

In recent years, with the popularity of songs classified as "[mumble rap](https://en.wikipedia.org/wiki/Mumble_rap)" there has been less of a focus in this regard, leading some fans of "true hip-hop" to look down upon the musicians currently in the spotlight. Part of the criticism of modern rap can often center around how forgettable the lyrics and songs are due to their simplistic nature. 

While learning about *recurrent neural networks* (RNNs)I came across a [video](https://www.youtube.com/watch?v=ZMudJXhsUpY) by Laurence Moroney explaining how AI can be used to generate poetry after training upon a corpus of Irish poems. This sparked an idea to try and do the same with modern rap lyrics. I specifically decided to choose this genre because of my familiarity with it and thinking that the songs may be more wordy since the artists often are not singing any melodies.

To add a twist, I wanted to go about my data collection in an intentionally biased manner in this case so that the results may mimic the source material as much as possible. However, I will try to maintain some sort of randomness in the generated text, so overfitting wil be a big concern during training.

## Packages
* requests==2.23.0
* regex==2020.6.8
* pickle==0.7.4
* pandas==0.23.4
* numpy==1.16.5
* tensorflow==2.2.0
* seaborn==0.9.0
* matplotlib==3.0.2
* nltk==3.4.5
* beautifulsoup4==4.7.1
* importlib-metadata==0.23
* custom functions (see functions.py in repository)

## Datatset Creation
#### Gathering the Data
In order to get all of the lyrics together as one package, I needed to scrape in two steps:
1. Get the links to the songs themselves
2. Get the songs via the links from Step 1

Step 1 can be completed using the BeautifulSoup4 API and a for-loop. Step 2 required a custom function that scraped songs and stored them in a time-delayed manner, allowing me to avoid DDoS filters. In total I was able to collect 180 songs to train the RNN with.
#### Prepping the Data
Some of the songs I collected included other artists in the lyrics, so I found a way using BeautifulSoup4 to separate the lyrics within a song depending on who was speaking. This was in an effort to ensure that the *only* lyrics the RNN sees are directly from Tyga, although I did train another model using the full collection of lyrics. 

Roughly speaking my process for the EDA was as follows:
1. Separate out lyrics without any featured artists by song title using a for-loop
2. Remove any non-Tyga lyrics via the headers using `lyric_header_checker()` and `lyric_header_filter()`
3. Split the lyrics into a list of strings line by line according to the song structure using `lyric_line_splitter()`
4. Remove punctuations, symbols, and lowercase all the lyrics using a regular expression within `song_cleaner()`
5. Tokenize the lyrics using NLTK's `word_tokenize()`

## EDA
#### All Solo Songs
asdfasdfas
#### N-Grams
qerqwetwe
#### Stopwords Out
aqeiruwq
#### Statistics
qpoeiru

## Creating the RNN
qoeiuroqiuerq
#### Architecture
rtoieurert
#### Training Strategy
poiuoie

## Text Generation
poiuoiuiopiuj
#### Prep Work
qoeiure
#### Censorship
eoirupoq

## Results
qeoriuqpoeiur
#### Sreamlit App
poieuporqwe
#### Future Work
asdfasdfd