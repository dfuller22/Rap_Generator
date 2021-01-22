def album_aggregator(soup_obj):
    ## Collects songs into dictionary based upon album titles

    import regex

    ## Creating container
    album_dict = {}

    ## Going through each tag in soup
    for tag in soup_obj:
        
        ## Storing tag's class
        class_ = tag.attrs['class'][0]
        
        ## Check for album title
        if class_ == 'album':
            ## Try: To store results created later in loop
            ## Exc: Print marking beginning of album_dict 
            try:
                album_dict[title_cln] = mid_dict
            except:
                print('1st Album!')
            
            ## Storing album title + regex out title in quotes ("")
            title = tag.text
            title_mid = regex.findall(r'(\".+\")', title)
            
            ## Try: Pop title from regex list + remove quotes
            ##      + set title as key + mid_dict for storing songs
            ## Exc: Print marking quirk in 'other song' album tag
            try:
                title_mid = title_mid.pop()
                title_cln = title_mid[1:-1]
                album_dict[title_cln] = None
                mid_dict = {}
            except IndexError:
                print('Empty list!')
        
        ## Check for song title/link
        elif class_ == 'listalbum-item':
            ## Pull song title
            song_title = tag.text
            ## Pull out link + regex/pop out extra chararacters
            song_conts = tag.contents[0]
            song_link = song_conts.attrs['href']
            song_link = regex.findall(r'(\/lyrics.+)', song_link).pop()
            ## Store link in mid_dict by song title
            mid_dict[song_title] = song_link

    return album_dict

def legendary_album_splitter(input_dict):
    ## Fixes error in legendary album representing 'other songs'

    ## Container for 'other songs' + index helper for 'for' loop
    other_holder = {}
    idxer = list(range(len(input_dict['Legendary'])))

    ## Going over songs in 'Legendary' album, removing songs 
    ## that should be in 'other' category
    for idx, key in zip(idxer, input_dict['Legendary']):

        ## Storing beginning of 'other' songs + set of beginning index
        if key == '40 Mill':
            other_holder[key] = input_dict['Legendary'][key]
            splitter = idx
        ## Pass song before '40 Mill'
        else:
            ## Try: check for idx past '40 Mill' + store song
            ## Exc: continue in loop until 'splitter' is defined
            try:
                ## Storing each song after '40 Mill'
                if idx > splitter:
                    other_holder[key] = input_dict['Legendary'][key]
            except NameError:
                pass

    ## Removing 'other' songs from original dictionary
    for key in other_holder:
        input_dict['Legendary'].pop(key)

    ## Storing 'other' songs from original dictionary + make a copy
    input_dict['Other Songs'] = other_holder
    output_dict = input_dict.copy()

    return output_dict

def song_scraper(dict_, names, limit=5, verbose=True):
    ## Scrapes songs on a randomized time delay stored in a dict

    import time
    import requests
    import numpy as np
    from bs4 import BeautifulSoup
    
    ## Create list to pull random intervals from + counter
    time_splits = np.linspace(10.129, 300.783, num=40)
    counter = 0
    skips = 0
    
    ## Setting limit to requests
    for title in names:
        ## Enforcing limit
        if counter >= limit:
            break
            
        ## Try: Join strings to create full song URL + Q.C.
        ## Except: Q.C. for non-str elements (already scraped)
        try:
            end_url = dict_[title]
            start_url = 'https://www.allthelyrics.com'
            full_url = start_url + end_url
            print(20*'--')
            print(f'Song to be scraped: {title}')
        except:
            ## Optional Q.C. of skipped songs
            if verbose:
                print(20*'**')
                print(f'Song already scraped: {title}')
                print(type(end_url))
                print(20*'**')
            ## Counter for skipped songs
            skips += 1
            continue
            
        ## Generating soup
        resp = requests.get(full_url)
        if resp.status_code == 200:            
            soup = BeautifulSoup(resp.text, 'html.parser')

            ## Collect lyrics 'div' from song soup as a bs4 tag + store in song dict
            song_lyrics = soup.findAll('div', attrs={'class': 'content-text-inner'}).pop()
            dict_[title] = song_lyrics

            ## Increase counter + random sampling for sleep time between requests
            counter += 1
            alarm = np.random.choice(time_splits)
            rounding = np.random.choice(list(range(2,6)))
            print(f'Sleeping {round(alarm, 2)} seconds...')
            time.sleep(round(alarm, rounding))
            print('Ding!')
            print(20*'--')
            
        else:
            print(f'Something wrong with link for {title}')
            continue
    
    ## Q.C. for skipped songs
    if not verbose:
        print(f'Total number of songs skipped: {skips}')

def song_scraping_stats(dict_):
    ## Helps to determine how many songs need to be scraped

    ## Result container
    status_dict = {'tag': 0, 'string': 0}
    
    ## Iterate over dict + check if link (str)
    for key in dict_:
        status = type(dict_[key])
        
        ## Count links else scraped lyrics (tag)
        if status == str:
            status_dict['string'] += 1
        else:
            status_dict['tag'] += 1

    return status_dict

def lyric_header_checker(dict_, display_multi=True, sort=True):
    ## Helps to locate 'who' is rapping via headers; used in filtering lyrics

    import regex

    ## Setting variable for matching + results container
    nav_str_ = 'NavigableString'
    tag_res = {}

    ## Iterate over dict + store songs
    for key in dict_:
        song = dict_[key]

        ## Iterate over BeautifulSoup elements
        for lyric in song:
            type_ = type(lyric)

            ## Skipping elements w/o lyrics
            if type_.__name__ == nav_str_:
                continue

            ## Locating headers over song sections
            reg_res = regex.findall(r'\[.+\]', lyric.text)

            ## Removing single headers
            if len(reg_res) == 1:
                item = reg_res.pop()

                ## Adding to results per instance
                try:
                    tag_res[item] += 1
                except:
                    tag_res[item] = 1

            ## Optional display of multiple speakers
            elif len(reg_res) > 1:
                if display_multi:
                    print('--'*10)
                    print(reg_res)

            else:
                continue
    ## Sorting headers by highest number of occurrences
    if sort:
        fin_res = {k: v for k, v in sorted(tag_res.items(),
                                           key=lambda item:item[1],
                                           reverse=True)}

        return fin_res

def lyric_header_filter(dict_, to_drop, display_multi=True, sort=True):
    ## Locates and stored lyrics depending on exluding list 'to_drop'; determined from 'header_checker'
    import regex

    ## Setting variable for matching + results container
    nav_str_ = 'NavigableString'
    tag_res = {}

    ## Iterate over dict + store songs in variable + mid-results container
    for key in dict_:
        song = dict_[key]
        lyric_holder = []

        ## Iterate over BeautifulSoup elements
        for lyric in song:
            type_ = type(lyric)

            ## Skipping elements w/o lyrics
            if type_.__name__ == nav_str_:
                continue

            ## Locating headers over song sections
            reg_res = regex.findall(r'\[.+\]', lyric.text)

            ## Removing single headers
            if len(reg_res) == 1:
                item = reg_res.pop()
                
                ## Adding to results while excluding unwanted headers
                if item in to_drop:
                    pass
                else:
                    lyric_holder.append(lyric.text)

            ## Optional display of multiple speakers
            elif len(reg_res) > 1:
                if display_multi:
                    print('--'*10)
                    print(reg_res)
                
                ## Iterate over headers in lyric section
                for item in reg_res:
                    ## Only adding sections not in 'to_drop'
                    if item in to_drop:
                        pass
                    else:
                        lyric_holder.append(lyric.text)                    

            ## Adding lyrics w/o headers
            else:
                lyric_holder.append(lyric.text)

        ## Storing all lyric sections in song by title        
        tag_res[key] = lyric_holder
    
    ## Optional sorting by song title
    if sort:
        fin_res = {k: v for k, v in sorted(tag_res.items(),
                                           key=lambda item:item[1],
                                           reverse=True)}
        
        return fin_res

def lyric_line_splitter(lyric_dict):
    ## Splits blocks of lyrics into a list of str repre. song lines

    ## Results container
    result_dict = {}
    
    ## Iterate through each song
    for song_title in lyric_dict:
        ## Store copied lyrics in variable + set merge container
        song = lyric_dict[song_title].copy()
        song_merged = []
        
        ## Iterate through each grouping of lyrics + merge split results
        for lyric in song:
            song_merged.extend(lyric.splitlines())
            
        ## Store list of merged lyrics
        result_dict[song_title] = song_merged
        
    return result_dict

def clean_song(list_of_lyrics):
    ## Helper function designed for 'song_cleaner'

    import regex
    
    ## Results container
    cleaned = []

    ## Iterate of song lines in list
    for line in list_of_lyrics:
        
        ## Remove punctuation + symbols, add to results
        if not '[' in line and not ']' in line:
            clean_line = regex.sub(r'[^\w\s]', '', line)
            cleaned.append(clean_line.lower())
            
    return cleaned

def song_cleaner(dict_of_songs):
    ## Iterates over dictionary of lyrics, returns dict of lower/punct-less strs

    ## Results container
    results = {}
    
    ## Iterate over dictionary, create copy + store cleaned version
    for song_title in dict_of_songs:
        song_raw = dict_of_songs[song_title].copy()
        results[song_title] = clean_song(song_raw)
        
    return results

def tokenize_lyrics(list_of_lyrics):
    ## Joins + tokenizes list of strings (lyrics)

    from nltk import word_tokenize

    ## Join list of strings by whitespace then tokenize via NLTK
    lyrics_joined = ' '.join(list_of_lyrics)
    lyrics_tokens = word_tokenize(lyrics_joined)
    
    return lyrics_tokens

def lyric_tokenizer(dict_of_songs):
    ## Uses 'tokenize_lyrics' to iterate over dictionary of lyrics

    ## Results container
    results = {}
    
    ## Iterate over songs + make a copy
    for song_title in dict_of_songs:
        lyrics_raw = dict_of_songs[song_title].copy()
        ## Tokenize + store results
        lyrics_tokens = tokenize_lyrics(lyrics_raw)
        results[song_title] = lyrics_tokens
        
    return results

def series_ratio(series, keep_df=False):
    ## Returns a pandas series with values as a % of total; optional df keep

    import pandas as pd

    ## Store total for later use    
    total = series.values.sum()

    ## Set into DataFrame for manipulation + copy ratio series
    res_df = pd.DataFrame(series)
    res_df.columns = ['Values']
    res_df['Ratio'] = res_df['Values'] / total
    ratio_series = res_df['Ratio'].copy()
    
    ## Optional return selection
    if keep_df:
        return res_df
    else:
        return ratio_series

def freqdist_plotter(tokens, premade_fd=False, n_common=None, h_plot=False, normalize_plot=False, show_ratio=False, figsize=(10,10)):
    ## Helper function to plot token freqdists w/variety of options for display

    import nltk
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    ## Check for n_common + create FreqDist if necessary
    if isinstance(n_common, int):
        freqdist = pd.Series(dict(nltk.FreqDist(tokens).most_common(n_common)))
    elif isinstance(n_common, type(None)):
        freqdist = pd.Series(dict(nltk.FreqDist(tokens)))
    elif premade_fd:
        freqdist = pd.Series(tokens)
    else:
        return f"Wrong input {type(n_common)}, use 'int' or 'None'."
        
    ## Setting figure & ax for plots
    fig, ax = plt.subplots(figsize=figsize)
    
    ## Check to normalize FreqDist values
    if normalize_plot:
        ## Noramlize FreqDist
        percent_plot = series_ratio(freqdist)
        
        ## Cropping plot if selected for
        if n_common:
            percent_plot = percent_plot.sort_values(ascending=False).head(n_common)
        else:
            percent_plot = percent_plot.sort_values(ascending=False)

        ## Setting plot to horizontal if selected for    
        if h_plot:
            bar_plot = sns.barplot(x=percent_plot.values, y=percent_plot.index, orient='h', ax=ax)
        else:
            bar_plot = sns.barplot(x=percent_plot.index, y=percent_plot.values, ax=ax)

        ## Setting title    
        plt.title('Normalized Frequency Distribution')

    else:
        ## Setting non-normalized plot to horizontal if selected for
        if h_plot:
            bar_plot = sns.barplot(x=freqdist.values, y=freqdist.index, orient='h', ax=ax)
        else:
            bar_plot = sns.barplot(x=freqdist.index, y=freqdist.values, ax=ax)

        ## Setting title    
        plt.title('Frequency Distribution')
    
    ## Rotate xticks on vertical plots only
    if h_plot:
        plt.show();
    else:
        plt.xticks(rotation=30)
        plt.show();
    
    ## Check for info displays
    if show_ratio:
        
        ## Try to display premade percent_plot series
        try:
            type(percent_plot)
            print('**'*10)
            print(f'Top {n_common} tokens usage rate (%):')
            print('**'*10)
            display(percent_plot)
        ## Create percent_plot series if needed
        except NameError:
            ratios = series_ratio(freqdist)
            
            ## Crop ratio list if selected for
            if n_common:
                ratios_sorted = ratios.sort_values(ascending=False).head(n_common)
            else:
                ratios_sorted = ratios.sort_values(ascending=False)
            ## Display info!
            print('**'*10)
            print(f'Top {n_common} tokens usage rate (%):')
            print('**'*10)
            display(ratios_sorted)

def n_gram_creator(tokens, top_n=20, n=2, freq_filter=None, window_size=None, counts=False, show_freq=True, show_pmi=False, keep=None):
    # Helper function creating [2-4]grams with a variety of options
    
    import nltk.collocations as colloc
    from nltk import bigrams, trigrams
    
    ## Check if n-gram is supported
    if n in [2,3,4]:
        
        ## Allowing for non-contiguous ngram creation
        if isinstance(window_size, int):
            window = window_size
        else:
            window = n
        
        ## Bigram setup
        if n == 2:
            word = 'Bi'
            
            if counts:
                ngrams = bigrams(tokens)
                return ngrams
            else:
                ngram_measures = colloc.BigramAssocMeasures()
                ngram_finder = colloc.BigramCollocationFinder.from_words(tokens, window_size=window)

        ## Trigram setup    
        elif n == 3:
            word = 'Tri'
            
            if counts:
                ngrams = trigrams(tokens)
                return ngrams
            else:
                ngram_measures = colloc.TrigramAssocMeasures()
                ngram_finder = colloc.TrigramCollocationFinder.from_words(tokens, window_size=window)

        ## Quadgram setup
        elif n == 4:
            word = 'Quad'
            ngram_measures = colloc.QuadgramAssocMeasures()
            ngram_finder = colloc.QuadgramCollocationFinder.from_words(tokens, window_size=window)       

        ## Applying frequency filter to results if selected for    
        if isinstance(freq_filter, int):
            ngram_finder.apply_freq_filter(freq_filter)

        ## Create ngram scores    
        ngram_score = ngram_finder.score_ngrams(ngram_measures.raw_freq)
        ngram_pmi_score = ngram_finder.score_ngrams(ngram_measures.pmi)
        
        ## Optional display
        if show_freq:
            print(f'Top {top_n} {word}-grams by frequency')
            display(ngram_score[:top_n])
        
        ## Optional display
        if show_pmi:
            print(f'PMI score for {top_n} {word}-grams')
            display(ngram_pmi_score[:top_n])

        ## Optional return   
        if keep == 'score':
            return ngram_score
        elif keep == 'pmi':
            return ngram_pmi_score
    
    ## Messaging for non-supported ngrams
    else:
        return f"{n}-grams are not supported. Try 2, 3, or 4."

def token_stat_generator(tokens, fd_plot=True, fd_n_common=20, fd_normalize=False, fd_show_ratio=False, ngram=False, n=2, ngram_top_n=20, ngram_freq_filter=None, ngram_window=None, ngram_count=False, ngram_pmi=False):
    # Function that serves as one line implementation of FreqDist plotter/N-gram creator for display
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import nltk
    
    ## Enacting FreqDist plotter
    if fd_plot:
        freqdist_plotter(tokens, normalize_plot=fd_normalize, n_common=fd_n_common, show_ratio=fd_show_ratio)
    
    ## Enacting N-gram creator
    if ngram:
        n_gram_creator(tokens, top_n=ngram_top_n, n=n, freq_filter=ngram_freq_filter, window_size=ngram_window, counts=ngram_count, show_pmi=ngram_pmi)

def n_gram_plot_prepper(ngrams, reverse_sort=True, type_='freqdist', keep='join'):
    # Helper function that prepares ngrams to be used in 'freqdist_plotter'
    
    from nltk import FreqDist
    
    ## Creating FreqDist + sort values; join ngram tokens with '_'
    if type_ == 'freqdist':
        ngram_fd = FreqDist(ngrams)
        ngram_sorted = {k:v for k,v in sorted(ngram_fd.items(), key=lambda item:item[1], reverse=reverse_sort)}
        ngram_joined = {'_'.join(k):v for k,v in sorted(ngram_fd.items(), key=lambda item:item[1], reverse=reverse_sort)}
    ## Sort + join ngram tokens (FreqDist not needed)
    elif type_ == 'pmi':
        ngram_sorted = {item[0]:item[1] for item in sorted(ngrams, key=lambda item:item[1], reverse=reverse_sort)}
        ngram_joined = {'_'.join(k):v for k,v in sorted(ngrams, key=lambda item:item[1], reverse=reverse_sort)}
    
    ## Optional return for type of ngram
    if keep == 'join':
        return ngram_joined
    elif keep == 'sort':
        return ngram_sorted
    else:
        return f"Must set keep parameter to either 'join' or 'sort', not {keep}."

def string_concat(list_of_strings, quick_check=False, qc_rate=25, qc_amount=25):
    # Helper function that combines a list of strings into one string
    
    ## Results container + counter 
    results = []
    qc_countdown = 0
    
    ## Iterate over each string in list
    for orig_string in list_of_strings:
        
        ## Adding next string in list to the end of combined result
        ## Remove and reset results list
        try:
            new_string = results[0] + '\n' + orig_string.strip()
            results.pop()
            results.append(new_string)
        ## Create starting point
        except IndexError:
            results.append(orig_string.strip())

        ## Increment counter    
        qc_countdown += 1
        
        ## Optional display
        if quick_check and (qc_countdown == qc_rate):
            print(f'Last {qc_amount} strings joined:')
            display(results[-qc_amount:])
            qc_countdown = 0 
            
    return results

def dict_string_concat(dict_of_strings, quick_check=False, qc_rate=25, qc_amount=25):
    # Helper function that applies 'string_concat' over a dictionary of lists

    ## Midpoint container
    mid_results = []
    
    ## Iterate over keys, create copy of list, then extend midpoint results
    for key in dict_of_strings:
        list_of_strings = dict_of_strings[key].copy()
        mid_concat = string_concat(list_of_strings, quick_check=quick_check, qc_rate=qc_rate, qc_amount=qc_amount)
        mid_results.extend(mid_concat)
    
    ## Execution display
    print('***'*20)
    print(f'{len(mid_results)} strings pulled from dictionary')
    print('***'*20)
    
    ## Final merge of all songs
    fin_results = string_concat(mid_results, quick_check=quick_check, qc_rate=qc_rate, qc_amount=qc_amount)
    
    return fin_results

def song_stats(song):
    # Helper function that creates a list of total and unique word in a string

    ## Join the song into one string, then split by words
    song_split = ' '.join(song).split()
    
    ## Gather song length + total unique words
    num_words = len(song_split)
    num_unique = len(set(song_split))
    
    return [num_words, num_unique]

def song_stat_df_generator(dict_of_songs):
    # Applies 'song_stats' over a dictionary of lists containing lyrics. Returns DataFrame
    # of engineered stats for plotting

    import pandas as pd
    
    ## Container for results
    result_dict = {}
    
    ## Go through each song, collect + store stats
    for key in dict_of_songs:
        song = dict_of_songs[key].copy()
        stats = song_stats(song)
        result_dict[key] = stats
    
    ## Create dataframe with totals + unique words
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    ## Reset index to store title as column + set column names
    result_df.reset_index(inplace=True)
    result_df.columns = ['title', 'total_words', 'unique_words']
    
    ## Feature engineering using totals + unique words
    result_df['unique_total_ratio'] = result_df['unique_words'] / result_df['total_words']
    result_df['avg_total'] = result_df.aggregate('mean', axis=0)['total_words']
    result_df['avg_unique'] = result_df.aggregate('mean', axis=0)['unique_words']
    result_df['avg_unique_ratio'] = result_df.aggregate('mean', axis=0)['unique_total_ratio']
    
    return result_df

class Timer():
    
    """Timer class designed to keep track of and save modeling runtimes. It
    will automatically find your local timezone. Methods are .stop, .start,
    .record, and .now"""
    
    def __init__(self, fmt="%m/%d/%Y - %I:%M %p", verbose=None):
        import tzlocal
        self.verbose = verbose
        self.tz = tzlocal.get_localzone()
        self.fmt = fmt
        
    def now(self):
        import datetime as dt
        return dt.datetime.now(self.tz)
    
    def start(self):
        if self.verbose:
            print(f'---- Timer started at: {self.now().strftime(self.fmt)} ----')
        self.started = self.now()
        
    def stop(self):
        print(f'---- Timer stopped at: {self.now().strftime(self.fmt)} ----')
        self.stopped = self.now()
        self.time_elasped = (self.stopped - self.started)
        print(f'---- Time elasped: {self.time_elasped} ----')
        
    def record(self):
        try:
            self.lap = self.time_elasped
            return self.lap
        except:
            return print('---- Timer has not been stopped yet... ----')
        
    def __repr__(self):
        return f'---- Timer object: TZ = {self.tz} ----'

class Censor():
    
    def __init__(self, lyrics):
        self._lyrics = lyrics
        self._targeted = None
        self._token_set = None
        self.__censored_words = ['ass', 'asses', 'fuck', 'fucked', 'fucks', 'fuckin',
                        'fucking', 'motherfuck', 'motherfucker', 'motherfuckin',
                        'motherfucking', 'bitch', 'bitches', 'cock', 'dick',
                        'dicks', 'pussy', 'pussies', 'shit', 'shits', 'shitty',
                        'faggot', 'fag', 'fags', 'nigga', 'niggas', 'nigger']
        
    def create_set(self, show=False):
        ## Set creation + stat assignment
        self._token_set = list(set(self._lyrics))
        self.__token_num_total = len(self._lyrics)
        self.__token_num_unique = len(set(self._token_set))
        
        ## Optional Q.C.
        if show:
            print(f'Total tokens: {self.__token_num_total:,}')
            print(f'Total unique tokens: {self.__token_num_unique:,}')
            
    def find_targets(self, extra_targets=None):
        ## Check if additional censors needed
        if isinstance(extra_targets, list):
            self.__censored_words.extend(extra_targets)
        
        ## Check for smaller version of lyrics
        if self._token_set:
            targets_within = []

            ## Collecting only the words applicable to these lyrics
            for word in self.__censored_words:
                if word in self._token_set:
                    targets_within.append(word)
            
            ## Assignment of lyric-specific curses
            self._targeted = targets_within
        
        ## Kick out if not done yet
        else:
            print(10*'--', 'Error', 10*'--')
            return 'Please use .create_set() first..'
            
    def count_targets(self, lyric_type='I', show=False):
        results = {}
        
        ## Check for necessary info
        if self._targeted == None:
            print(10*'--', 'Error', 10*'--')
            return 'Please use .find_targets() first..'
        
        ## Type of lyrics to count
        if lyric_type == 'I':
            lyrics2count = self._lyrics
        elif lyric_type == 'A':
            lyrics2count = self._altered_lyrics
        elif lyric_type == 'C':
            lyrics2count = self._cleaned_lyrics
        elif lyric_type == 'M':
            lyrics2count = self._muted_lyrics
        else:
            return "Please use 'I', 'A', 'C', or 'M' for 'lyric_type' parameter. See docstring for more info."
        
        ## Begin counting
        for word in lyrics2count:
            ## Pull curses and increment
            if word in self._targeted:
                try:
                    results[word] += 1
                except KeyError:
                    results[word] = 1
        
        ## Sort + assign to attribute
        self.__target_counts = {k:v for k,v in sorted(results.items(), key=lambda item:item[1], reverse=True)}
        
        ## Optional display
        if show:
            return self.__target_counts
        
    def alter_targets(self, replacements):
        ## Set copy for manipulation
        self._altered_lyrics = self._lyrics.copy()
        
        ## Check for necessary info + create dict to map values
        try:
            trgt2cnsrd = dict.fromkeys(self._targeted, None)
        except NameError:
            print(10*'--', 'Error', 10*'--')
            return 'Please use .find_targets() first..'
        
        ## Check replacements match dictionary
        assert len(trgt2cnsrd) == len(replacements), "Make sure 'replacement' list is equal in length to targeted words"
        
        ## Matching up dictionary to replacement list
        for i, key in enumerate(trgt2cnsrd):
            trgt2cnsrd[key] = replacements[i]
            
        ## Altering strings
        ## Iter. over each target/replacement
        for key, val in trgt2cnsrd.items():
            ## Iter. over each lyric token
            for i, token in enumerate(self._altered_lyrics):
                ## Regex to sub token inplace + make_copy check
                if key == token:
                    self._altered_lyrics[i] = val
        
    def mute_targets(self, replacement='*'):
        ## Set copy for manipulation
        self._muted_lyrics = self._lyrics.copy()
       
        ## Iter. over each target word
        for target in self._targeted:
            ## Iter. over each lyric token
            for i, token in enumerate(self._muted_lyrics):
                ## Assigning replacement per target size
                if target == token:
                    if len(token) == 1:
                        self._muted_lyrics[i] = replacement
                    elif len(token) == 2:
                        self._muted_lyrics[i] = token[0] + replacement
                    elif len(token) == 3:
                        token = token[0] + replacement*2
                        self._muted_lyrics[i] = token
                    elif len(token) >= 4:
                        token = token[0] + replacement*(len(token)-2) + token[-1]
                        self._muted_lyrics[i] = token
                    else:
                        print('Length of <=0!')
        
    def remove_targets(self):
        ## Set copy for manipulation
        self._cleaned_lyrics = self._lyrics.copy()
        
        ## Iter. over each target token
        for target in self._targeted:
            ## Iter. over each lyric token
            for token in self._cleaned_lyrics:
                ## Remove each match to avoid errors
                if token == target:
                    self._cleaned_lyrics.remove(target)
        
    def add_target(self, to_add, multi_add=False, show=False):
        ## Iter. over each new target OR append .targeted
        if multi_add:
            for new_trgt in to_add:
                if new_trgt in self._targeted:
                    continue
                else:
                    self._targeted.append(new_trgt)
        else:
            if to_add in self._targeted:
                return f"'{to_add}' already a target!"
            else:
                self._targeted.append(to_add)
        
        ## Optional display
        if show:
            return self._targeted

    def get_lyrics(self, lyric_type='I'):
        ## Type of lyrics to return
        if lyric_type == 'I':
            return self._lyrics
        elif lyric_type == 'A':
            return self._altered_lyrics
        elif lyric_type == 'C':
            return self._cleaned_lyrics
        elif lyric_type == 'M':
            return self._muted_lyrics
        else:
            return "Please use 'I', 'A', 'C', or 'M' for 'lyric_type' parameter. See docstring for more info."

    def get_word_counts(self, lyric_type='I'):
        ## Result container
        results = {}
        
        ## Type of lyrics to count
        if lyric_type == 'I':
            lyrics2count = self._lyrics
        elif lyric_type == 'A':
            lyrics2count = self._altered_lyrics
        elif lyric_type == 'C':
            lyrics2count = self._cleaned_lyrics
        elif lyric_type == 'M':
            lyrics2count = self._muted_lyrics
        else:
            return "Please use 'I', 'A', 'C', or 'M' for 'lyric_type' parameter. See docstring for more info."

        ## Begin counting
        for word in lyrics2count:
            ## Find words and increment
                try:
                    results[word] += 1
                except KeyError:
                    results[word] = 1
        
        ## Sort + assign to attribute
        word_counts = {k:v for k,v in sorted(results.items(), key=lambda item:item[1], reverse=True)}
        
        ## Display
        return word_counts
