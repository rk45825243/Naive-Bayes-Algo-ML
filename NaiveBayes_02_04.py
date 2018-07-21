
# coding: utf-8

# In[1]:


from sklearn import naive_bayes


# In[2]:


import numpy as np


# In[3]:


df=np.loadtxt(r"C:\Users\admin\Desktop\wd_ml_02_04_20_jun_18\spambase.data.txt",dtype="str",delimiter=",")
df


# In[4]:


# import pandas as pd


# In[5]:


# pd.read_csv(r"C:\Users\admin\Desktop\wd_ml_02_04_20_jun_18\spambase.data.txt")


# In[6]:


y=df[:,-1]


# In[7]:


y=y.astype('int')
y.shape


# In[8]:


x=df[:,:-1]
x.shape


# In[9]:


x=x.astype('float')


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[12]:


nm=naive_bayes.MultinomialNB()


# In[13]:


nm.fit(x_train,y_train)


# In[14]:


nm.score(x_test,y_test)*100


# In[15]:


count=0
for i in range(len(x_test)):
    p=nm.predict([x_test[i]])
    if(p==y_test[i]):
        count+=1
print("Accuracy score is %.2f%%"%((count/len(x_test))*100))


# In[16]:


PREDICT_INDEX=69
p=nm.predict([x_test[PREDICT_INDEX]])
print("Actual result:",y_test[PREDICT_INDEX])
print("Predicted result:",p)


# In[17]:


import pandas as pd


# In[18]:


df=pd.read_csv(r"C:\Users\admin\Desktop\wd_ml_02_04_20_jun_18\smsspamcollection_2\SMSSpamCollection",sep="\t",names=["status","messages"])
df.head()


# In[19]:


df.info()


# In[20]:


df.index


# In[21]:


df.head()


# In[22]:


df.loc[df['status']=='spam',"status"]=0
df.loc[df['status']=='ham',"status"]=1


# In[23]:


df.head(2)


# In[24]:


y=df['status'].values
y


# In[25]:


x=df['messages']
x.head()


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer


# In[27]:


arr=["Go until jurong point point","U dun say so early hor"]


# In[28]:


count_vt=CountVectorizer()


# In[29]:


a=count_vt.fit(arr)


# In[30]:


tr_arr=count_vt.transform(arr)


# In[31]:


tr_arr.toarray()


# In[32]:


count_vt.get_feature_names()


# In[33]:


ct_vt=CountVectorizer()


# In[34]:


a=ct_vt.fit(x)


# In[35]:


obj_vt=ct_vt.transform(x)


# In[36]:


arr_vt=obj_vt.toarray()
arr_vt


# In[37]:


arr_vt.shape


# In[47]:


y=y.astype("int")
y


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(arr_vt,y,test_size=0.2)


# In[49]:


print(len(x_train),x_train.shape)


# In[50]:


print(len(x_test),x_test.shape)


# In[51]:


print(len(y_train),y_train.shape)


# In[52]:


print(len(y_test),y_test.shape)


# In[53]:


nbm=naive_bayes.MultinomialNB()


# In[54]:


nbm.fit(x_train,y_train)


# In[58]:


count=0
for i in range(len(x_test)):
    p=nbm.predict([x_test[i]])
    if(y_test[i]==p):
        count+=1
print("Accuracy is %5.2f%%"%((count/len(x_test))*100))


# In[59]:


nbm.score(x_test,y_test)


# In[61]:


# ct_vt.get_feature_names()


# In[62]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[63]:


y


# In[65]:


x.head(2)


# In[66]:


tfidf=TfidfVectorizer()


# In[67]:


from sklearn.linear_model import LogisticRegression


# In[68]:


from sklearn.datasets import load_iris


# In[70]:


data=load_iris()
y=data['target']
y[:5]


# In[71]:


x=data['data']
x[:5]


# In[72]:


llr=LogisticRegression()


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[75]:


llr.fit(x_train,y_train)


# In[79]:


PREDICT_INDEX=17
p=llr.predict([x_test[PREDICT_INDEX]])
print("Actual result is: ",y_test[PREDICT_INDEX])
print("Predicted result is: ",p)


# In[80]:


llr.score(x_test,y_test)


# In[87]:


count=0
for i in range(len(x_test)):
    p=llr.predict([x_test[i]])
    if(y_test[i]==p):
        count+=1
print("Accuracy score is %5.2f%%"%((count/len(x_test))*100))

