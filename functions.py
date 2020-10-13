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