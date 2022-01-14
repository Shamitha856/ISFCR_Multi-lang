#!/usr/bin/env python
# coding: utf-8

# # Fake News Detection

# In[109]:


pwd


# In[110]:


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


# In[111]:


df_fake = pd.read_csv("E:\\isfcr project\\Fake.csv")
df_true = pd.read_csv("E:\\isfcr project\\True.csv")


# In[112]:


df_fake


# In[113]:


df_true


# In[114]:


df_fake.head(10)


# In[115]:


df_true.head(10)


# In[116]:


df_fake["real/fake"] = "Fake"
df_true["real/fake"] = "Real"


# In[117]:


df_fake.shape, df_true.shape


# In[118]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23448, 23438, -1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
#inplace=true coz v do not want another dataset to be returned, axis=0=>along a row


# In[119]:


df_fake.shape, df_true.shape


# In[120]:


df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("E:\\isfcr project\\manual_testing.csv")
#axis=0 indicates concatenation along series or rows


# In[121]:


df_merge = pd.concat([df_fake, df_true], axis = 0)
df_merge.head(10)


# In[122]:


df = df_merge.sample(frac = 1)
#sample() is a function from pandas, used to generate sample random row or column from the function(to shuffle)
#frac indicates the length of dataframe values to be returned , 1 indicates to return all values


# In[123]:


df.head(10)


# In[124]:


df


# In[125]:


df.shape


# In[126]:


df.isnull().sum()


# In[127]:


def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
#to remove unusual and irregular expressions in the text column
#sub is an built-in library to replace substring with another substring


# In[128]:


df["text"] = df["text"].apply(word_drop)


# In[129]:


df.head(100)


# In[130]:


df.shape


# In[131]:


def word(subject):
    subject = subject.lower()
    return subject
#converts all data in column subject to lowercase


# In[132]:


df["subject"] = df["subject"].apply(word)


# In[133]:


df.head(10)


# In[134]:


df.shape


# In[135]:


df.isnull().sum()


# In[136]:


df = df.drop_duplicates(subset = ['text','title','date'], keep = 'last').reset_index(drop = True)
#when drop=false(default value) a new dataframe is returned
#keep=last indicates that while dropping the duplicates last one is kept
#function of pandas


# In[137]:


df


# In[138]:


df.shape


# In[139]:


df["date"]=df.date.str.replace(' ','-')


# In[140]:


df.head(10)


# In[141]:


df["date"]=df.date.str.replace(',','')


# In[142]:


df.head(100)


# In[143]:


df["date"]=df.date.str.replace(r'[-]$','', regex=True)
#regex=true indicates to replace for all regular expressions


# In[144]:


df.head(10)


# In[145]:


df["date"]=df.date.str.replace("January",'1')
df["date"]=df.date.str.replace("February",'2')
df["date"]=df.date.str.replace("Febraury",'2')
df["date"]=df.date.str.replace("March",'3')
df["date"]=df.date.str.replace("April",'4')
df["date"]=df.date.str.replace("May",'5')
df["date"]=df.date.str.replace("June",'6')
df["date"]=df.date.str.replace("July",'7')
df["date"]=df.date.str.replace("August",'8')
df["date"]=df.date.str.replace("September",'9')
df["date"]=df.date.str.replace("October",'10')
df["date"]=df.date.str.replace("November",'11')
df["date"]=df.date.str.replace("December",'12')


# In[146]:


df.head(1000)


# In[147]:


df["date"]=df.date.str.replace("Jan",'1')
df["date"]=df.date.str.replace("Feb",'2')
df["date"]=df.date.str.replace("Mar",'3')
df["date"]=df.date.str.replace("Apr",'4')
df["date"]=df.date.str.replace("Jun",'6')
df["date"]=df.date.str.replace("Jul",'7')
df["date"]=df.date.str.replace("Aug",'8')
df["date"]=df.date.str.replace("Sep",'9')
df["date"]=df.date.str.replace("Oct",'10')
df["date"]=df.date.str.replace("Nov",'11')
df["date"]=df.date.str.replace("Dec",'12')


# In[148]:


df.head(100)


# In[149]:


import datetime as dt


# In[150]:


df.info()
#prints the datatypes of all the columns


# In[151]:


df["date"]= pd.to_datetime(df["date"])


# In[152]:


df.sort_values(by='date', inplace=True)
#not to return a new dataset


# In[153]:


df.head(10)


# In[154]:


df = df.reset_index(drop=True)
#when drop=false(default) a new dataframe is returned


# In[155]:


df


# In[156]:


final_df = df.to_csv("E:\\isfcr project\\final_df.csv")


# In[157]:


#text-independent(x) real/fake-dependent(y)
x = df["text"]
y = df["real/fake"]


# In[158]:


#splitting data into train and test sets out of which 0.2 percent of data is for testing
#train_test_split is a function in sklearn for splitting data sets into two sub-arrays randomly 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[159]:


x_train


# In[160]:


x_test


# In[161]:


y_train


# In[162]:


y_test


# In[163]:


#text column has raw data which cant be used for computations so vectorize it into  vectors
from sklearn.feature_extraction.text import TfidfVectorizer


# In[164]:


#stop words are english words which doesn't add much meaning to a sentence and can be safely removed
#term frequency–inverse document frequency
tfvect = TfidfVectorizer(stop_words='english', max_df = 0.7)
#max_df=0.7=>ignore terms that appear in more than 70% of the texts
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)
#fit_tranform returns (sentence_index,feature_index) count.....fit_transform does some calculation and then transforms
#it removes all 0 entries in a sparse matrix
#transform just tranforms the raw text into number


# In[165]:


tfid_x_train


# In[166]:


tfvect


# In[167]:


tfid_x_test


# In[168]:


#this model remains passive for normal inputs(true) and turns aggressive on different input(fake)
classifier = PassiveAggressiveClassifier(max_iter=50)
#max_iter=maximum number of passes over the training data
classifier.fit(tfid_x_train,y_train)
#fit-taining part of the modelling process


# In[169]:


y_pred = classifier.predict(tfid_x_test)
#accuracy_score -> sklearn
score = accuracy_score(y_test,y_pred)
#accuracy between the predicted and actual values
print(f'Accuracy: {round(score*100,2)}%')


# In[170]:


cf = confusion_matrix(y_test,y_pred, labels=['Fake','Real'])
print(cf)
#2 rows and 2 columns that reports the number of false positives,false negatives,true positives and true negatives
#actual fake and actual real along rows and actual fake and actual true along columns
#the classifier predicted (19+4143) news as real and (3553+19) as fake, in reality (4143+19) news are real and (3553+51)are fake
#'True -ve', 'False +ve'
#'False -ve', 'True +ve'


# In[171]:


import pylab as pl
pl.matshow(cf)
pl.title("Confusion matrix of the classifier")
pl.colorbar()
pl.show()


# In[172]:


import matplotlib as plt
import seaborn as sns
group_names = ['True -ve', 'False +ve', 'False -ve', 'True +ve']
group_percentages = ["{0:.3%}".format(value) for value in
                    cf.flatten()/np.sum(cf)]
labels = [f"{v1}\n\n{v2}"
          for v1,v2 in
          zip(group_names, group_percentages)]
labels = np.asarray(labels).reshape(2,2)
#ax = plt.axes()
ax=sns.heatmap(cf, annot = labels, fmt = '')
ax.set_title('Confusion Matrix for PassiveAggressiveClassifier')


# In[173]:


print(len(tfvect.vocabulary_))
#Total no of vocabularies identified


# In[174]:


#Pickle in Python is primarily used in serializing and deserializing a Python object structure.
#In other words, it's the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.
import pickle


# In[175]:


#pickle.dump(pac,open('model_1.pkl', 'wb'))-direct method
with open('model_english_pac.pkl','wb') as handle:
    pickle.dump(classifier,handle,protocol=pickle.HIGHEST_PROTOCOL)
#handle-file object returned after opening the model
#pickle.HIGHEST_PROTOCOL indicates the highest version of pickle


# In[176]:


print(len(tfvect.vocabulary_))


# In[177]:


pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nbmodel', MultinomialNB())])
#multinomial naive bayes algorithm is a probabilistic learning method used in NLP


# In[178]:


pipeline.fit(x_train, y_train)


# In[179]:


score = pipeline.score(x_test, y_test)
print('Accuracy', score)
#score-returns a score or a loss


# In[180]:


pred = pipeline.predict(x_test)


# In[181]:


print(classification_report(y_test, pred))
#to measure the quality of predictions
#precisions-measure of classifier's exactness
#recall-measure of classifier's completeness
#f1-score=weighted harmonic mean of precision and recall
#support-number of actual occurences of class in specified dataset


# In[182]:


print(len(tfvect.vocabulary_))


# In[183]:


print(confusion_matrix(y_test, pred))
cf_pipe= confusion_matrix(y_test, pred)


# In[184]:


import seaborn as sns
import matplotlib as plt
import seaborn as sns
group_names = ['True -ve', 'False +ve', 'False -ve', 'True +ve']
group_percentages = ["{0:.3%}".format(value) for value in
                    cf_pipe.flatten()/np.sum(cf_pipe)]
labels = [f"{v1}\n\n{v2}"
          for v1,v2 in
          zip(group_names, group_percentages)]
labels = np.asarray(labels).reshape(2,2)
#ax = plt.axes()
ax=sns.heatmap(cf_pipe, annot = labels, fmt = '')
ax.set_title('Confusion Matrix for Pipeline Model')


# In[185]:


import pickle


# In[186]:


with open('model_english_pipeline.pkl','wb') as handle:
    pickle.dump(pipeline,handle,protocol=pickle.HIGHEST_PROTOCOL)


# In[187]:


#Logistic Regression, to model the probability of certain class or event existing
from sklearn.linear_model import LogisticRegression


# In[188]:


LR = LogisticRegression()
LR.fit(tfid_x_train,y_train)
#fits the training dataset to the object


# In[189]:


#given a trained model it predicts the new set of data,it returns learned label of each object in an array
pred_lr=LR.predict(tfid_x_test)


# In[190]:


#Returns the co-efficient of the determination of the prediction-model accuracy level
LR.score(tfid_x_test, y_test)


# In[191]:


#comparison between dataset class and predicted values
print(classification_report(y_test, pred_lr))


# In[192]:


print(confusion_matrix(y_test, pred_lr))


# In[193]:


#Decision tree classification
from sklearn.tree import DecisionTreeClassifier


# In[194]:


DT = DecisionTreeClassifier()
DT.fit(tfid_x_train, y_train)


# In[195]:


pred_dt = DT.predict(tfid_x_test)


# In[196]:


DT.score(tfid_x_test, y_test)


# In[197]:


print(classification_report(y_test, pred_dt))


# In[198]:


print(confusion_matrix(y_test, pred_dt))


# In[199]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier


# In[200]:


GBC = GradientBoostingClassifier(random_state=0)
#random_state=0 controls the verbosity(fact and quality of using more words than needed) while splitting and prediction
GBC.fit(tfid_x_train, y_train)


# In[201]:


pred_gbc = GBC.predict(tfid_x_test)


# In[202]:


GBC.score(tfid_x_test, y_test)


# In[203]:


print(classification_report(y_test, pred_gbc))


# In[204]:


print(confusion_matrix(y_test, pred_gbc))


# In[205]:


#Random forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[206]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(tfid_x_train, y_train)


# In[207]:


pred_rfc = RFC.predict(tfid_x_test)


# In[208]:


RFC.score(tfid_x_test, y_test)


# In[209]:


print(classification_report(y_test, pred_rfc))


# In[210]:


print(confusion_matrix(y_test, pred_rfc))


# In[211]:


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
    new_def_test["text"] = new_def_test["text"].apply(word_drop) 
    new_x_test = new_def_test["text"]
    new_xv_test = tfvect.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    pred_pac = classifier.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {} \nPAC Prediction: {} ".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0]),
                                                                                                              output_lable(pred_pac[0])))


# In[212]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:




