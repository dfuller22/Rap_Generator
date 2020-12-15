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

def sdfa