def lite_web_sentiment_estimator(url):
    #RECEIVING AND PARSING WEBPAGE TEXT
    import requests
    html = requests.get(url).text #gets and stores webpage's hmtl source code in string variable 'html'
    
    from bs4 import BeautifulSoup as bs
    soup = bs(html) #converts string variable to unicode
    souped_text = soup.get_text() #stores unicode text in variable 'souped_text'
    encoded_text = souped_text.encode('utf-8') #converts unicode text to utf-8
    
    import nltk
    tokens = nltk.word_tokenize(encoded_text) #tokenizes unicode text, separating each word and turning it into an item in list 'tokens'
    alpha = [word for word in tokens if word.isalpha() == True] #filters tokens to exclude non-alphabetic words
    
    import pandas as pd
    alpha_df = pd.DataFrame(alpha)
    alpha_df.columns=['Word']

    df = alpha_df
    
    #ESTABLISHING WORD COUNTS FOR EACH UNIQUE WORD
    df['Word Count'] = [1 for x in df['Word']]
    df = df.groupby(['Word']).count()
    df = df.reset_index()
    
    #DETERMINING FREQUENCY-WEIGHTS FOR EACH WORD
    df['Frequency-Weight'] = 1
    df.ix[df['Word Count'] == 1, 'Frequency-Weight'] = 2
    df.ix[df['Word Count'] > 10, 'Frequency-Weight'] = 0.5
        
    #CALCULATING SHARE OF TEXT FOR EACH UNIQUE WORD
    df['Share of Words'] = df['Word Count']/df['Word Count'].sum()
    
    #ASSIGNING INITAL POLARITY WITH PRE-DEFINED POS/NEG WORDBANKS
    df['Polarity'] = 0
    
    #DEFINING CUSTOM WORDBANKS
    positive_emotions = ['curious', 'attract', 'surprise', 'hope', 'Hope', 
                     'thank', 'joy', 'relief', 'proud', 'Joy', 
                     'generous', 'sympath', 'love', 'amuse', 'Love',
                     'delight', 'elat', 'excit', 'happy', 'happi', 'Happi',
                     'joy', 'pleasure', 'affection', 'empath', 'friendl']
    negative_emotions = ['alarm', 'disgust', 'indifferen', 'fear', 'Fear', 
                     'rage', 'sorrow', 'grief', 'frustrat', 'disappoint', 
                     'embarrass', 'shame', 'guilt', 'remorse', 'greed', 'Greed',  
                     'miser', 'jealous', 'cruel', 'hate', 'anger', 'annoyed', 'Anger', 'Hate',
                     'disgust', 'irrit', 'anxious', 'anxiety', 'helpless',
                     'worry', 'doubt', 'shame', 'bored', 'despar', 'hurt']
 
    #PROCESSING ADDITIONAL PRE-DEFINED POSITIVE AND NEGATIVE WORDBANKS
    pos_bank = 'https://raw.githubusercontent.com/c-trl/nlp-with-xanga-entries/master/positive_wordbank.txt'
    pos_words_text = requests.get(pos_bank).text
    pos_soup = bs(pos_words_text) #converts string variable to unicode
    souped_pos_words = pos_soup.get_text() #stores unicode text in variable 'souped_text'
    encoded_pos_words = souped_pos_words.encode('utf-8') #converts unicode text to utf-8
    positive_words = nltk.word_tokenize(encoded_pos_words) #tokenizes unicode text, separating each word and turning it into an item in list 'tokens'

    neg_bank = 'https://raw.githubusercontent.com/c-trl/nlp-with-xanga-entries/master/negative_wordbank.txt'    
    neg_words_text = requests.get(neg_bank).text
    neg_soup = bs(neg_words_text) #converts string variable to unicode
    souped_neg_words = neg_soup.get_text() #stores unicode text in variable 'souped_text'
    encoded_neg_words = souped_neg_words.encode('utf-8') #converts unicode text to utf-8
    negative_words = nltk.word_tokenize(encoded_neg_words)    
    
    df.ix[df.Word.isin(positive_emotions), 'Polarity'] = 1
    df.ix[df.Word.isin(negative_emotions), 'Polarity'] = -1
    df.ix[df.Word.isin(positive_words), 'Polarity'] = 1
    df.ix[df.Word.isin(negative_words), 'Polarity'] = -1
    
    df['Adjusted Sentiment'] = df['Polarity'] * df['Frequency-Weight']
    df['Relative Weight'] = df['Word Count'] * df['Adjusted Sentiment']
    
    print
    print 'Text source:',url
    print 'Weight of all positive words:', sum(df[[x > 0 for x in df['Relative Weight']]].sort('Relative Weight', ascending=False)['Relative Weight'])
    print 'Weight of all negative words:', sum(df[[x < 0 for x in df['Relative Weight']]].sort('Relative Weight', ascending=False)['Relative Weight'])

    x = sum(df['Relative Weight'])
    if x > 0:
        pos_or_neg = 'Positive'
    if x < 0:
        pos_or_neg = 'Negative'
    print 'Aggregate text sentiment:', pos_or_neg
    print
