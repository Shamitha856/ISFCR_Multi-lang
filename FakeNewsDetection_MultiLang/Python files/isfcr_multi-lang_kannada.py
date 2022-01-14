#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
import re
import string
import pickle


# In[64]:


df_true = pd.read_csv("E:\\isfcr project\\true_kannada.csv")
df_true1 = pd.read_csv("E:\\isfcr project\\true1_kannada.csv")
df_true2 = pd.read_csv("E:\\isfcr project\\true2_kannada.csv")
df_true3 = pd.read_csv("E:\\isfcr project\\true3_kannada.csv")


# In[65]:


pd.set_option('display.max_colwidth', None)


# In[66]:


df_true1 = df_true1.drop(['text_url'], axis = 1)
df_true2 = df_true2.drop(['text_url'], axis = 1)
df_true3 = df_true3.drop(['text_url'], axis = 1)


# In[67]:


df_true.head(10)


# In[68]:


df_true.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[69]:


df_true1.head(10)


# In[70]:


df_true2.head(10)


# In[71]:


df_true3.head(10)


# In[72]:


df_true.shape, df_true1.shape, df_true2.shape, df_true3.shape


# In[73]:


df_true = df_true.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_true1 = df_true1.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_true2 = df_true2.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_true3 = df_true3.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)


# In[74]:


df_true.shape, df_true1.shape, df_true2.shape, df_true3.shape


# In[75]:


df_merge_true = pd.concat([df_true, df_true1, df_true2, df_true3], axis = 0)


# In[76]:


df_merge_true["real/fake"] = "Real"


# In[77]:


df_merge_true.shape


# In[78]:


df_merge_true = df_merge_true.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)


# In[79]:


df_merge_true.tail(10)


# In[80]:


df_merge_true.shape


# In[81]:


df_fake = pd.read_csv("E:\\isfcr project\\fake_kannada.csv")
df_fake1 = pd.read_csv("E:\\isfcr project\\fake1_kannada.csv")
df_fake2 = pd.read_csv("E:\\isfcr project\\fake2_kannada.csv")
df_fake3 = pd.read_csv("E:\\isfcr project\\fake3_kannada.csv")
df_fake4 = pd.read_csv("E:\\isfcr project\\fake4_kannada.csv")
df_fake5 = pd.read_csv("E:\\isfcr project\\fake5_kannada.csv")
df_fake6 = pd.read_csv("E:\\isfcr project\\fake6_kannada.csv")
df_fake7 = pd.read_csv("E:\\isfcr project\\fake7_kannada.csv")
df_fake8 = pd.read_csv("E:\\isfcr project\\fake8_kannada.csv")


# In[82]:


df_fake1 = df_fake1.drop(['text_url'], axis = 1)
df_fake2 = df_fake2.drop(['text_url'], axis = 1)
df_fake3 = df_fake3.drop(['text_url'], axis = 1)
df_fake4 = df_fake4.drop(['text_url'], axis = 1)
df_fake5 = df_fake5.drop(['text_url'], axis = 1)
df_fake6 = df_fake6.drop(['text_url'], axis = 1)
df_fake7 = df_fake7.drop(['text_url'], axis = 1)
df_fake8 = df_fake8.drop(['text_url'], axis = 1)


# In[83]:


df_fake.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[84]:


df_fake.head(10)


# In[85]:


df_fake1.head(10)


# In[86]:


df_fake2.tail(10)


# In[87]:


df_fake3.head(10)


# In[88]:


df_fake4.head(10)


# In[89]:


df_fake5.head(10)


# In[90]:


df_fake6.head(10)


# In[91]:


df_fake7.head(10)


# In[92]:


df_fake8.head(10)


# In[93]:


df_fake.shape, df_fake1.shape, df_fake2.shape, df_fake3.shape, df_fake4.shape, df_fake5.shape, df_fake6.shape, df_fake7.shape, df_fake8.shape


# In[94]:


df_fake = df_fake.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake1 = df_fake1.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake2 = df_fake2.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake3 = df_fake3.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake4 = df_fake4.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake5 = df_fake5.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake6 = df_fake6.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake7 = df_fake7.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)
df_fake8 = df_fake8.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)


# In[95]:


df_fake.shape, df_fake1.shape, df_fake2.shape, df_fake3.shape, df_fake4.shape, df_fake5.shape, df_fake6.shape, df_fake7.shape, df_fake8.shape


# In[96]:


df_merge_fake = pd.concat([df_fake, df_fake1, df_fake2, df_fake3, df_fake4, df_fake5, df_fake6, df_fake7, df_fake8], axis = 0)


# In[97]:


df_merge_fake["real/fake"] = "Fake"


# In[98]:


df_merge_fake.shape


# In[99]:


df_merge_fake = df_merge_fake.drop_duplicates(subset = ['text_name'], keep = 'last').reset_index(drop = True)


# In[100]:


df_merge_fake.tail(10)


# In[101]:


df_merge = pd.concat([df_merge_fake, df_merge_true], axis = 0)


# In[102]:


df = df_merge.sample(frac = 1)


# In[103]:


df.head(10)


# In[104]:


final_df_kannada = df.to_csv("E:\\isfcr project\\final_df_kannada.csv")


# In[105]:


df.shape


# In[106]:


STOPS_LIST = ['ಈ','ಆದರೆ','ಎಂದು','ಅವರ','ಮತ್ತು','ಎಂಬ','ಅವರು','ಬಗ್ಗೆ','ಆ','ಇದೆ','ಇದು','ನಾನು','ನನ್ನ','ಅದು','ಮೇಲೆ','ಈಗ','ಹಾಗೂ','ಇಲ್ಲ',
'ನನಗೆ','ಅವರಿಗೆ','ತಮ್ಮ','ಮಾಡಿ','ನಮ್ಮ','ಮಾತ್ರ','ದೊಡ್ಡ','ಅದೇ','ಯಾವ','ಆಗ','ತುಂಬಾ','ನಾವು','ದಿನ','ಬೇರೆ','ಅವರನ್ನು','ಎಲ್ಲಾ','ನೀವು',
'ಹೊಸ','ಮುಂದೆ','ಹೇಗೆ','ನಂತರ','ಇಲ್ಲಿ','ಅಲ್ಲ','ಬಳಿಕ','ಹಾಗಾಗಿ','ಒಂದೇ','ಅದನ್ನು','ಬಂದ','ನಿಮ್ಮ','ಇತ್ತು','ಹೇಳಿ','ಮಾಡಿದ','ಅದಕ್ಕೆ','ಆಗಿ',
'ಎಂಬುದು','ಅಂತ','ಕೆಲವು','ಮೊದಲು','ಬಂದು','ಇದೇ','ನೋಡಿ','ಕೇವಲ','ಎರಡು','ಇನ್ನು','ಅಷ್ಟೇ','ಎಷ್ಟು','ಚಿತ್ರದ','ಮಾಡಬೇಕು','ಹೀಗೆ','ಕುರಿತು',
'ಉತ್ತರ','ಎಂದರೆ','ಇನ್ನೂ','ಮತ್ತೆ','ಏನು','ಪಾತ್ರ','ಮುಂದಿನ','ಸಂದರ್ಭದಲ್ಲಿ','ಮಾಡುವ','ವೇಳೆ','ನನ್ನನ್ನು','ಅಥವಾ','ಜೊತೆಗೆ','ಹೆಸರು']


# In[107]:


x = df["text_name"]
y = df["real/fake"]


# In[108]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[109]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[110]:


type(x_train)


# In[111]:


from nltk.tokenize import word_tokenize

word_tokens={}
filtered_sentence = []

for x in x_train:
    x=re.sub("[a-zA-Z]"," ",str(x))
    word_tokens[x] = word_tokenize(x)
    #print(word_tokens[x])
    for i in word_tokens[x]:
        if i in STOPS_LIST:
            word_tokens[x].remove(i)
    print(word_tokens[x])
    
'''from inltk.inltk import tokenize

hindi_text = """ರೈತ ಪ್ರತಿಭಟನೆ ಬೆನ್ನಲ್ಲೇ ಮುಖೇಶ್ ಅಂಬಾನಿ ಜೊತೆ ಪಂಜಾಬ್ ಸಿಎಂ ಮಾತುಕತೆ..?"""

# tokenize(input text, language code)
tokenize(hindi_text, "kn")'''


# In[112]:


type(word_tokens)


# In[113]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[114]:


word_tokens = pd.Series(word_tokens)


# In[115]:


type(word_tokens)


# In[116]:


tfvect = TfidfVectorizer()
tfid_x_train = tfvect.fit_transform(x_train.values.astype('U'))
tfid_x_test = tfvect.transform(x_test.values.astype('U'))


# In[117]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[118]:


#this model remains passive for normal inputs(true) and turns aggressive on different input(fake)
classifier = PassiveAggressiveClassifier(max_iter=50)
#max_iter=maximum number of passes over the training data
classifier.fit(tfid_x_train,y_train)
#fit-taining part of the modelling process


# In[119]:


y_pred = classifier.predict(tfid_x_test)
#accuracy_score -> sklearn
score = accuracy_score(y_test,y_pred)
#accuracy between the predicted and actual values
print(f'Accuracy: {round(score*100,2)}%')


# In[120]:


cf = confusion_matrix(y_test,y_pred, labels=['Fake','Real'])
print(cf)


# In[121]:


import pickle

#pickle.dump(pac,open('model_1.pkl', 'wb'))-direct method
with open('model_kannada_pac.pkl','wb') as handle:
    pickle.dump(classifier,handle,protocol=pickle.HIGHEST_PROTOCOL)
#handle-file object returned after opening the model
#pickle.HIGHEST_PROTOCOL indicates the highest version of pickle


# In[122]:


#Manual Testing
def output_lable(n):
    if n == "Fake":
        return "Fake News"
    elif n == "Real":
        return "Real News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    #creates a dataframe
    new_def_test["text"] = new_def_test["text"]
    new_x_test = new_def_test["text"]
    new_xv_test = tfvect.transform(new_x_test)

    pred_pac = classifier.predict(new_xv_test)
    return print("\n\nPAC Prediction:  ", output_lable(pred_pac[0]))


# In[123]:


news = str(input())
manual_testing(news)


# In[ ]:




