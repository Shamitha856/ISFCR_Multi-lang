#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd


# In[73]:


df_true = pd.read_csv("E:\\isfcr project\\true_news.csv")
df_fake = pd.read_csv("E:\\isfcr project\\fake_news.csv")


# In[74]:


df_true.head()


# In[75]:


df_fake.head()


# In[76]:


df_fake.shape, df_true.shape


# In[77]:


df_fake.isnull().sum()


# In[78]:


df_true.isnull().sum()


# In[79]:


#df = pd.DataFrame(df)


# In[80]:


df_true.drop(['Unnamed: 0', 'full_title','short_description'], axis=1, inplace=True)


# In[81]:


df_fake.drop(['Unnamed: 0', 'full_title','short_description'], axis=1, inplace=True)


# In[82]:


for i in list(df_true):
    df_true[i]=df_true[i].str.replace('|', '')
    df_true[i]=df_true[i].str.replace('?', '')
    df_true[i]=df_true[i].str.replace(':', '')
    df_true[i]=df_true[i].str.replace(';', '')
    df_true[i]=df_true[i].str.replace("'", '')
    df_true[i]=df_true[i].str.replace('"', '')
    df_true[i]=df_true[i].str.replace(',', '')
    df_true[i]=df_true[i].str.replace('.', '')
    df_true[i]=df_true[i].str.replace('(', '')
    df_true[i]=df_true[i].str.replace(')', '')
    df_true[i]=df_true[i].str.replace('\n', '')
    df_true[i]=df_true[i].str.replace('&', '')


# In[83]:


for i in list(df_fake):
    df_fake[i]=df_fake[i].str.replace('|', '')
    df_fake[i]=df_fake[i].str.replace('?', '')
    df_fake[i]=df_fake[i].str.replace(':', '')
    df_fake[i]=df_fake[i].str.replace(';', '')
    df_fake[i]=df_fake[i].str.replace("'", '')
    df_fake[i]=df_fake[i].str.replace('"', '')
    df_fake[i]=df_fake[i].str.replace(',', '')
    df_fake[i]=df_fake[i].str.replace('.', '')
    df_fake[i]=df_fake[i].str.replace('(', '')
    df_fake[i]=df_fake[i].str.replace(')', '')
    df_fake[i]=df_fake[i].str.replace('\n', '')
    df_fake[i]=df_fake[i].str.replace('&', '')


# In[84]:


suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
}

def hi_stem(word):
    for L in 5, 4, 3, 2, 1:
        if len(word) > L + 1:
            for suf in suffixes[L]:
                if word.endswith(suf):
                    return word[:-L]
    return word


# In[85]:


id=list(df_fake.index)
id1=list(df_true.index)


# In[86]:


str_temp=""
count=0
for i in list(df_fake):
    count=0
    for j in list(df_fake[i]):
        for words in j.split():
            str_temp+=hi_stem(words)
            str_temp+=" "
        df_fake.loc[id[count],i]=str_temp
        str_temp=""
        count+=1


# In[87]:


str_temp=""
count=0
for i in list(df_true):
    count=0
    for j in list(df_true[i]):
        for words in j.split():
            str_temp+=hi_stem(words)
            str_temp+=" "
        df_true.loc[id1[count],i]=str_temp
        str_temp=""
        count+=1


# In[88]:


'''def deEmojify(inputString):
    returnString = ""
    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            returnString += ''
    return returnString'''


# In[89]:


import re
import sys

def remove_emoji(string):
    emoji_pattern = re.compile("["
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
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# In[90]:


def processText(text):
    text = text.lower()
    text = re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
    text = re.sub('@[^s]+','',text)
    text = re.sub('[s]+', ' ', text)
    text = re.sub(r'#([^s]+)', r'1', text)
    text = re.sub(r'!', r'', text)
    text = text.strip('"')
    return text


# In[91]:


#df_fake["long_description"] = df_fake["long_description"].apply(deEmojify)


# In[92]:


df_fake["long_description"] = df_fake["long_description"].apply(remove_emoji)


# In[93]:


df_fake["long_description"] = df_fake["long_description"].apply(processText)


# In[94]:


#df_true["long_description"] = df_true["long_description"].apply(deEmojify)


# In[95]:


df_true["long_description"] = df_true["long_description"].apply(remove_emoji)


# In[96]:


df_true["long_description"] = df_true["long_description"].apply(processText)


# In[97]:


df_fake.head(5)


# In[98]:


df_true.to_csv('true_news.csv')
df_fake.to_csv('fake_news.csv')


# In[99]:


'''#@title other useful code
len(true_news), len(fake_news)
fake_news = fake_news.head(len(true_news))
len(true_news), len(fake_news)

# assign labels
fake_news['label'] = 1
true_news['label'] = 0

# join true and false dataset
news = pd.concat([fake_news, true_news])
from sklearn.utils import shuffle
news = shuffle(news)'''


# In[100]:


df_fake["real/fake"] = "Fake"
df_true["real/fake"] = "Real"


# In[101]:


df_merge = pd.concat([df_fake, df_true], axis = 0)
df_merge.head(10)


# In[102]:


df = df_merge.sample(frac = 1)
#sample() is a function from pandas, used to generate sample random row or column from the function(to shuffle)
#frac indicates the length of dataframe values to be returned , 1 indicates to return all values


# In[103]:


df.head()


# In[104]:


df = df.drop_duplicates(subset = ['long_description'], keep = 'last').reset_index(drop = True)
#when drop=false(default value) a new dataframe is returned
#keep=last indicates that while dropping the duplicates last one is kept
#function of pandas


# In[105]:


pd.set_option('display.max_colwidth', None)


# In[106]:


df.head()


# In[107]:


final_df_hindi = df.to_csv("E:\\isfcr project\\final_df_hindi.csv")


# In[108]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[109]:


x = df["long_description"]
y = df["real/fake"]


# In[110]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[111]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[112]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[113]:


STOPS_LIST = ['हें', 'है', 'हैं', 'हि', 'ही', 'हो', 'हे', 'से', 'अत', 'के', 'रहे', 'का', 'की', 'कि', 'तो', 'ने', 'एक',
              'नहीं', 'पे', 'में', 'वाले', 'सकते', 'वह', 'वे', 'कई', 'होती', 'आप', 'यह', 'और', 'एवं', 'को', 'मे', 'दो', 
              'थे', 'यदि', 'उनके', 'थी', 'पर', 'इस', 'साथ', 'लिए', 'जो', 'होता', 'या', 'लिये', 'द्वारा', 'हुई', 'जब', 'होते', 
              'व', 'न', 'उनकी', 'आदि', 'सकता', 'उनका', 'इतयादि', 'इतना', 'जिस', 'उस', 'कैसे', 'हूँ', 'ना', 'कहि', 'सम',
              'र्', 'कहँ', 'बस', 'अपना', 'यही', 'कहीं', 'हाँ', 'मैंने', 'जहँ', 'सब', 'यह', 'था', 'तुम', 'ये', 'जे', 'भी', 'हम',
              'अब', 'ऐसे', 'वहाँ', 'क्या', 'ओर', 'इसी', 'सके', 'कभी', 'हर', 'मेरी', 'कम', 'सा', 'उन्हें', 'मेरे', 'उन', 'कुछ',
              'इन', 'ऐसा', 'जहा', 'तीन']


# In[114]:


print(STOPS_LIST)


# In[115]:


x_train.shape


# In[116]:


#for i in x_train:
    #print(i)


# In[117]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

word_tokens={}
filtered_sentence = []

for x in x_train:
    word_tokens[x] = word_tokenize(x)
    #print(word_tokens[x])
    
    for i in word_tokens[x]:
        if i in STOPS_LIST:
            word_tokens[x].remove(i)
    print(word_tokens[x])


# In[118]:


import langid
for i in word_tokens:
    for x in word_tokens[i]:
        print(x)
        if(len(x)>3):
            lang = langid.classify(x)
            print(lang)
            try:
                if(lang!='hi' and lang!='en' and lang!='mr' and lang!='ne'):
                        word_tokens[i].remove(x)
            except:
                    print('error')


# In[119]:


tfvect = TfidfVectorizer()
tfid_x_train = tfvect.fit_transform(word_tokens)
tfid_x_test = tfvect.transform(x_test)


# In[120]:


tfid_x_train


# In[121]:


#this model remains passive for normal inputs(true) and turns aggressive on different input(fake)
classifier = PassiveAggressiveClassifier(max_iter=50)
#max_iter=maximum number of passes over the training data
classifier.fit(tfid_x_train,y_train)
#fit-taining part of the modelling process


# In[122]:


y_pred = classifier.predict(tfid_x_test)
#accuracy_score -> sklearn
score = accuracy_score(y_test,y_pred)
#accuracy between the predicted and actual values
print(f'Accuracy: {round(score*100,2)}%')


# In[123]:


cf = confusion_matrix(y_test,y_pred, labels=['Fake','Real'])
print(cf)


# In[124]:


import pickle

#pickle.dump(pac,open('model_1.pkl', 'wb'))-direct method
with open('model_hindi_pac.pkl','wb') as handle:
    pickle.dump(classifier,handle,protocol=pickle.HIGHEST_PROTOCOL)
#handle-file object returned after opening the model
#pickle.HIGHEST_PROTOCOL indicates the highest version of pickle


# In[125]:


def word_drop(text):
    text = text.str.replace('|','') 
    text = text.str.replace('?','')
    text = text.str.replace(':','')
    text = text.str.replace(';','')
    text = text.str.replace("'",'')
    text = text.str.replace('"','')
    text = text.str.replace(',','')
    text = text.str.replace('.','')
    text = text.str.replace('(','')
    text = text.str.replace(')','')
    text = text.str.replace('\n','')
    text = text.str.replace('&','')
    return text
#to remove unusual and irregular expressions in the text column
#sub is an built-in library to replace substring with another substring


# In[126]:


#Manual Testing
def output_lable(n):
    if n == "Fake":
        return "Fake News"
    elif n == "Real":
        return "Real News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news).apply(word_drop)
    #creates a dataframe
    new_def_test["text"] = new_def_test["text"]
    new_x_test = new_def_test["text"]
    new_xv_test = tfvect.transform(new_x_test)

    pred_pac = classifier.predict(new_xv_test)
    return print("\n\nPAC Prediction:  ", output_lable(pred_pac[0]))


# In[127]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:




