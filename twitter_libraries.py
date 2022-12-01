# -*- coding: utf-8 -*-

#base libraries
import tweepy
import pandas as pd
import numpy as np
import datetime
import re
#Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#Model apply
from sklearn.feature_extraction.text import CountVectorizer
#Run R code
from IPython import get_ipython
pd.options.display.max_colwidth = 80
import pickle
from dotenv import load_dotenv
load_dotenv()
import os 

def US_state_abrev(to_abreviation = True,
                   vector_or_list = False,
                   fips = False):

    if fips:
        value = 'FIPS'
    else:
        value = 'Name'
        
    if not to_abreviation:
         zip_db = pd.read_csv(r'data/codes.csv')
         zip_dict = zip_db.set_index('Postal Code')[value].to_dict()  
         
         if not vector_or_list:
             return zip_dict
         else:
             change_abreviation = lambda t: zip_dict[t]
             vfunc = np.vectorize(change_abreviation)
             return vfunc
    else:
        zip_db = pd.read_csv(r'data/codes.csv')
        zip_dict = zip_db.set_index('Name')['Postal Code'].to_dict()
    
        if not vector_or_list:
            return zip_dict
        else:
            change_abreviation = lambda t: zip_dict[t]
            vfunc = np.vectorize(change_abreviation)
            return vfunc

us_dict = US_state_abrev()

# =============================================================================
#%% Authentication and extraction ACCOUNT USER functions
# =============================================================================

def authentication(): 

    consumerKey = os.getenv("CONSUMER_KEY")
    consumerSecret = os.getenv("CONSUMER_SECRET")
    accessToken = os.getenv("ACCESS_TOKEN")
    accessTokenSecret = os.getenv("ACCESS_TOKEN_SECRET")
    
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth, 
                    wait_on_rate_limit = True
                     )
    return api

#%% Extraction from twitter NEWS FEED 

def query_twitter(searchTerms, certain_time_ago = None, number_of_tweets = None, location = None):
    
    # searchTerms = "billion"
    # certain_time_ago = None
    # number_of_tweets = None
    # location = " geocode:49.25208599342474,-123.03391418427883,100000km"
    
    api = authentication()
    
    #place = api.geo_search(query="US", granularity="country")
    #place_id = place[0].id
    #searchTerm ="pizza"    
    if location:
        q_loc = location
    else:
        q_loc = ""
    
    if certain_time_ago:    
        print("Query with until")
        query ='{} until:{} -filter:links -filter:retweets -filter:replies{}'. \
            format(searchTerms, certain_time_ago, q_loc)
    else:
        print("Query without until")
        query ='{} -filter:links -filter:retweets -filter:replies{}'. \
            format(searchTerms, q_loc)
    print("This is the QUERY:",query)

    
    all_tweets = []
    
    new_tweets = []
    print("...Waiting for API to allow more calls")
    new_tweets = api.search(q = query, tweet_mode = 'extended',
                                    lang = 'en', count = 100, 
                                    #geocode = "49.25208599342474,-123.03391418427883,100000km"
                                    )
    #print(len(new_tweets))
    
    all_tweets.extend(new_tweets)
    
    try:
        oldest = all_tweets[-1].id - 1
        
        while len(new_tweets) > 0:
          
          print("getting tweets before %s" % (oldest))
          print("...Waiting for API to allow more calls")
          new_tweets = api.search(q = query, tweet_mode = 'extended', max_id = oldest,
                                        lang = 'en', count = 100,
                                        #geocode = "49.25208599342474,-123.03391418427883,100000km"
                                        )
          all_tweets.extend(new_tweets)
          oldest = all_tweets[-1].id - 1
          print("...%s tweets downloaded so far" % (len(all_tweets)))
          ### END OF WHILE LOOP ###
          
          if number_of_tweets:
              if len(all_tweets) > number_of_tweets:
                  break
          
        
        id_ = []
        date = []
        location = []
        screen_name = []
        name = []
        description = []
        tweets_text = []
        retweet_count = []
        source = []
        favourites_count = []
        followers_count = []
        friends_count = []
        profile_background_color = []
        verified = []
        
        i = 0
        for status in all_tweets:
            status = status.__dict__
            # print(status)
            try: 
                tweets_text.append(status['text'])
            except:
                tweets_text.append(status['full_text'])
                
            id_.append(status['id'])
            date.append(status['created_at'])
            retweet_count.append(status['retweet_count'])
            source.append(status['source'])
            
            if status['place']:    
                place = status['place'].__dict__
                location.append(place['full_name']+" - "+place['country_code'])
            else:
                location.append('not_determined')
                
            user = status['user'].__dict__
            
            screen_name.append(user['screen_name'])
            name.append(user['name'])
            description.append(user['description'])
            favourites_count.append(user['favourites_count'])
            followers_count.append(user['followers_count'])
            friends_count.append(user['friends_count'])  
            profile_background_color.append(user['profile_background_color'])
            verified.append(user['verified'])
            
            i +=1
            
        print("LENGTH LIST",len(tweets_text))
        
        dict_ = {'id':id_,'date':date, 'location': location, 'username':screen_name,'name':name,
                   'description':description, 'tweet':tweets_text, 'retweet_count':retweet_count,
                   'source':source,'favourites_count':favourites_count, 'followers_count':followers_count,
                   'friends_count':friends_count, 'profile_background_color':profile_background_color,
                   'friends_count':friends_count,'profile_background_color':profile_background_color,'verified':verified}
        
        df = pd.DataFrame(dict_)
        
        df.insert(2, 'search','{}'.format(searchTerms))
        return df
    except:
        df = pd.DataFrame()
     
def create_df(name, random_food, days_ago = None, number_of_tweets = None, location = None):
    
    today = datetime.date.today()
    
    if days_ago:
        certain_time_ago = today - datetime.timedelta(days=days_ago)
        certain_time_ago = str(certain_time_ago)
        print('...bringing tweets until',certain_time_ago)
    else:
        certain_time_ago = None
        
    final_data = pd.DataFrame(data = None, columns = ['date', 'location',
                                                      'username', 'search', 
                                                      'tweet'])
    
    if type(random_food) == list:
        # print('A LIST')
        for food in random_food:

            iter_data = query_twitter(food, certain_time_ago, number_of_tweets, location)
                           
            final_data = pd.concat([final_data, iter_data], axis = 0)
            print("iteration is in this keyword:",food)
            print(len(final_data))
        # final_data.to_csv('{}.csv'.format(name), index = False)
        return final_data
    else:
        # print('A STRING')
        iter_data = query_twitter(random_food, certain_time_ago, number_of_tweets, location)
        print(len(final_data))       
        # iter_data.to_csv('{}.csv'.format(name), index = False)
        return iter_data

#%% Clean functions

# function to remove emojis from text
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# function to clean text data
def clean_text(data):
    # remove user names
    data['tweet'] = [re.sub('@[\w]+','',x) for x in data['tweet']]
    # case conversion
    data['tweet'] = [x.lower() for x in data['tweet']]
    # Remove emojis
    
    data['tweet'] = [remove_emojis(text) for text in data['tweet']]

    # Remove url from text
    data['tweet'] = [re.sub(r'http\S+', '', text) for text in data['tweet']]

    # Remove # from text
    data['tweet'] = [re.sub(r'#', '', text) for text in data['tweet']]

    # Remove lead & trail spaces
    data['tweet'] = [text.lstrip() for text in data['tweet']]
    data['tweet'] = [text.strip() for text in data['tweet']]

    # Remove punctuation
    data['tweet'] = data['tweet'].map(lambda x: re.sub('[,\.!?]', '', x))
    
    data['date'] = pd.to_datetime(data['date'])
    
    return data

#%%Filtering
def get_word_withtin_target_sent_non_nlp(text, subject_names):

    list_ = []
    text = str(text)
    for subject_name in subject_names:
        subject_name = str(subject_name)
        if subject_name in text:
            list_.append(subject_name)   
            
    return list_

#%%TOPIC analysis

def get_metrics(path):    

    get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
    
    get_ipython().\
        run_cell_magic('R', '', 
                       '''
                       \n#Required Packages\n\n
                       
                       library(tidyverse)\n
                       library(tidytext)\n
                       library(topicmodels)\n
                       library(stm)\n
                       library(furrr)\n
                       library(reshape2)\n
                       library(tsne)\n
                       library(writexl)''')
    
    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n
                       #Directory\n\n
                                              
                       #Open Data\n
                       
                       tweets<- read.csv({},
                                         encoding="UTF-8", 
                                         header=T, 
                                         na.strings=c("","NA"))'''.format(path))
    
    get_ipython().run_cell_magic('R', '', '''\n
                                 #Clean.\n\n
                                 tweets <-tweets %>% mutate(tweet=str_replace_all(tweets$tweet, "@[a-zA-Z0-9_]{0,15}", ""))\n\ntweets <-tweets %>% mutate(tweet=str_replace_all(tweets$tweet, "#\\\\S+", ""))\n\ntweets <-tweets %>% mutate(tweet=str_replace_all(tweets$tweet, "[^\\x01-\\x7F]", " "))\n\ntweets <-tweets %>% mutate(tweet=str_replace_all(tweets$tweet, "http.+ |http.+$", " "))\ntweets <-tweets %>% mutate(tweet=str_replace_all(tweets$tweet, "http[[:alnum:]]*", " "))\n\ntweets <-tweets %>% mutate(tweet=str_replace_all(tweets$tweet, "[:digit:]", ""))\n\n#Drop duplicates\n\ntweets<-tweets %>% \n  distinct(tweet,.keep_all = T)\n\n#Number of words per tweet\n\ntweets<-tweets %>% mutate(words=str_count(tweet,"\\\\w+")) %>% as_tibble()\n\n#Filter tweets with less than 5 words\n\ntweets<-tweets %>% filter(words>=5)%>%select(-(words))\n\n\n#Tokenize data.\n\ntopics<-tweets %>% mutate(id=1:n()) %>% \n  unnest_tokens(word, tweet) %>% count(id,word,sort=TRUE)\n\n#Removing stopwords\n\ncustom_stop_words <- bind_rows(tibble(word = c("amp","lt","gt","it’s","i’ve","i’m","i’d",\n                                               "i’ll","hey","https","t.co","i’ma","imma","im",\n                                               "i\'ve","i\'m","i\'ve","i’m","dont","ve","gonna",\n                                               "ll","don"),  \n                                      lexicon = c("custom")), stop_words)\n\ncustom_stop_words1 <-custom_stop_words %>% mutate(word=str_replace(custom_stop_words$word, "\\\'", "’"))\n\ncustom_stop_words2 <-custom_stop_words %>% mutate(word=str_replace(custom_stop_words$word, "’", ""))\n\ncustom_stop_words3<-bind_rows(custom_stop_words,custom_stop_words1,custom_stop_words2)\n\n\ntopics<- topics %>%\n  anti_join(custom_stop_words3) %>% as_tibble()\n''')
   
    get_ipython().\
        run_cell_magic('R', '', 
                       '''library(ldatuning)''')
    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n
                       # Transform toe DocumentTermMatrix\n
                       #Use cast_dtm to tranform\n\n
                       
                       dtm <- topics %>%\n
                           cast_dtm(id, word, n)''')
    
    #Takes usually around 5 min
    get_ipython().run_cell_magic('R', '', '''\n
                                  #Number of Topics\n\n
                                 
                                 
                                  result <- FindTopicsNumber(\n  
                                                            dtm,\n  
                                                            topics = seq(from = 2, to = 25, by = 1),\n  
                                                            metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),\n  
                                                            method = "Gibbs",\n  
                                                            control = list(seed = 77),\n  
                                                            mc.cores = 2L,\n  
                                                            verbose = TRUE\n
                                                            )''')
                                  
    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n
                       
                       #Table Results\n\n
                       result''')
    
    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n
                       #Plot Metrics\n\n
                       
                       result %>% pivot_longer(!topics,names_to = "metric", values_to = "values")%>% ggplot(aes(x=topics,y=values))+geom_line()+\n  
                           facet_wrap(~ metric, scales = "free")+theme_light()\n\n
                           
                       #The lower CaoJuan 2009 and Griffiths 2004 metrics the better.\n
                       #Whereas, Arun2010 and Deveaud2014 The higher the better. Resources:\n
                       #explicando cada metrica.\n\n
                       #Arun: http://doi.org/10.1007/978-3-642-13657-3_43\n
                       #Cao: http://doi.org/10.1016/j.neucom.2008.06.011\n
                       #Devenaud: http://doi.org/10.3166/dn.17.1.61-84\n
                       #Griffiths: http://doi.org/10.1073/pnas.0307752101\n\n
                       
                       #Determine number optimal number of topics by chossing right chart\n
                       #Then send parameter to determine topic''')

def determine_topic(number_of_topics):
    
    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n
                       # Run model with n topics. \n\n
                       ap_lda <- LDA(dtm, k = {}, method= "Gibbs",
                                     control = list(seed = 576,alpha=2))'''.format(number_of_topics))

    get_ipython()\
        .run_cell_magic('R', '', 
                        '''\n
                        ap_topics <- tidy(ap_lda, matrix = "beta")\n
                        ap_topics\n\n
                        
                        #Most common terms per topic\n\n
                        
                        ap_top_terms <- ap_topics %>%\n
                        group_by(topic) %>%\n
                        top_n(10, beta) %>%\n
                        ungroup() %>%\n
                        arrange(topic, -beta)\n\n
                        ap_top_terms %>%\n
                        mutate(term = reorder_within(term, beta, topic)) %>%\n  
                        ggplot(aes(beta, term, fill = factor(topic))) +\n  
                        geom_col(show.legend = FALSE) +\n  
                        facet_wrap(~ topic, scales = "free") +\n
                        scale_y_reordered()\n''')

    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n## Probability of tweet within topic.\n\n
                       ap_documents <- tidy(ap_lda, matrix = "gamma")\n\n
                       x<-ap_documents %>% \n
                           group_by(document) %>%\n  
                               filter(gamma == max(gamma)) %>% arrange(document) %>% rename(id=document)\n\n
                       
                        tweets<-tweets %>% mutate(id=1:n()) \n\n
                        tweets<-tweets %>% mutate(id=as.factor(id))\n\n
                        
                        #Join data\n\n
                        
                        tweets1<-inner_join(x,tweets)''')

def allocate_topics_and_export(topics, output_path):
    
    counter = 1
    input_topic = []
    
    for topic in topics:
        
        item = 'topic== {} ~ "{}"'.format(counter, topic)
        counter = counter + 1
        input_topic.append(item)
    input_topic = ', '.join(input_topic)
    
    get_ipython().run_cell_magic('R', '', 
                                 '''\n
                                 #Allocate topic names\n\n
                                 tweets1<-tweets1 %>% \n  
                                 mutate(expected_topic = case_when(\n    
                                                                   {}
                                                                   ))'''.format(input_topic))

    get_ipython().\
        run_cell_magic('R', '', 
                       '''\n#Save into csv\n\n
                       
                       write_csv(tweets1,{})'''.format(output_path))

#%% GENDER & SENTIMENT

def apply_gender_model(df):
    
    cv = CountVectorizer(max_features = 2000)
    x = cv.fit_transform(df['tweet']).toarray()
#    print(x.shape)

    filenameDT = r'models\model_pt_f.sav'
    model = pickle.load(open(filenameDT, 'rb'))
    y_pred = model.predict(x)

    df = df.reset_index(drop = True)
    df = pd.concat([df,pd.DataFrame(y_pred, columns = ['gender'])], axis = 1)
    df.gender = df.gender.apply(lambda x: 'male' if x == 1 else 'female')
    
    return df

def get_sentiment(df):

    analyzer = SentimentIntensityAnalyzer()
    
    def define_sentiment(x):
        if x['neg'] > x['pos']:
            result = 'negative'
            return result
        elif x['pos'] > x['neg']:
            result = 'positive'
            return result
        elif x['pos'] == x['neg']:
            result = 'neutral'
            return result

    df['sentiment'] = df.tweet.apply(lambda x: analyzer.polarity_scores(x))   
    df.sentiment = df.sentiment.apply(lambda scores: define_sentiment(scores))
    
    return df

def get_state(x):
    result = x.split(' - ')[0]
    try:
        state =  result.split(', ')[1]
        if state == 'USA':
            result = result.split(', ')[0]
            try:
                return us_dict[result]
            except:
                return result
        else:
            return state
    except:
        return 'no_state'

