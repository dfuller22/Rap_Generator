def album_aggregator(soup_obj):

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
    
    status_dict = {'tag': 0, 'string': 0}
    
    for key in dict_:
        status = type(dict_[key])
        
        if status == str:
            status_dict['string'] += 1
        else:
            status_dict['tag'] += 1

    return status_dict