#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lexisnexis.dataset.news as news
import pandas  as pd
import csv
import re
import spacy
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[2]:


get_ipython().run_line_magic('store', '-r us_city_names')


# In[3]:


dataset = news.Dataset(cache_refresh=True)


# In[4]:


request = news.Request()
nlp = spacy.load('en_core_web_sm')


# In[5]:


response = dataset.get()


# In[6]:


df = response.data


# In[7]:


df["z_tuple"] = df.title.apply(lambda x: tuple(x))  
df = df.drop_duplicates(subset="z_tuple", keep="first")  


# In[8]:


df = df.reset_index()
del df['index']


# In[9]:


indicated_city = []
indicated_date = []
stated_city = []
title = []
content = []
corpus = []

    


for titles in df['title']:
    title.append(titles)
    
for cities in df['city']:
    indicated_city.append(cities)
    
for dates in df['publication_date']:
    indicated_date.append(dates)

for contents in df['content']:
    sents = []
    bean = []
    new_content = []
    words = contents.split(" ")
    for word in words:
        if word in us_city_names:
            bean.append(word)
        new_word = word.translate({ord("\'"):None}).replace(u'\xa0', u'').replace('p.m.', 'pm').replace('a.m.', 'am').title()
        new_content.append(new_word)
    stated_city.append(bean)
    s = " "
    contents = s.join(new_content)
    
    for texts in contents.split(". "):
        corpus.append(texts)
        if "Curfew" or "Curfews" in texts:
            sents.append(texts)
    content.append(sents)


# In[10]:


print(corpus[:100])


# In[11]:


data = pd.DataFrame({'Title': title, 'Indicated City': indicated_city, 'Date': indicated_date, 'Cleaned':content, 'Stated City':stated_city, 'Indicated City':indicated_city})
data


# In[12]:


'''
sentences_corpus = []
for i in range(len(data)):
    co = data['Cleaned'][i]
    for texts in co:
        block = []
        doc = nlp(texts)
        for token in doc:
            x = token.lemma_
            block.append(x)
        s = " "
        new_texts = s.join(block)
    sentences_corpus.append(new_texts)
    new_texts = ''
'''

filtered_sentences = []
for i in data['Cleaned']:
    block = []
    for sentence in i:
        words = sentence.split()
        for i in range(len(words)):
            if not words[i].isalpha():
                words[i] = ""
            #if words[i] in stopwords.words('english'):
                #words[i] = ""
            if words[i] in us_city_names:
                words[i] = ""
        s = " "
        filtered_sentence = s.join(words)
        block.append(' '.join(filtered_sentence.split()))
        filtered_sentence = ''
    filtered_sentences.append(block)
    
        
        


# In[13]:


print(len(filtered_sentences))


# In[92]:


table = pd.DataFrame({'Title': title, 'Indicated City': indicated_city, 'Date': indicated_date, 'Sample': filtered_sentences,'Stated City':stated_city, 'Indicated City':indicated_city})
table


# In[15]:


want = ["Pass", "Enact", "Impose", 'Imposed', "Enacted", "Passed"]
for i in table['Sample']:
    for z in i:
        for words in z.split(" "):
            if words in want:
                print(z)


# In[46]:


sample = {'New legislation was enacted in 1992, and a partnership was established between the Phoenix Police Department and the Department of Parks, Recreation, and Libraries (PRL)': '1', 'Chicago, Illinois Chicago passed its first curfew ordinance in July 1948': '1' , 'The organization already has taken on several curfew cases, including a 1995 Washington curfew and a San Diego curfew passed in 1947': '1', 'In June, the council passed a juvenile curfew ordinance modeled on one used in Dallas':'1' , 'In June, the council passed a juvenile curfew ordinance modeled on one used in DallasPasco commissioners passed a countywide curfew in March after more than 18 months of lobbying by Sheriff Lee Cannon When Zephyrhills passed its curfew ordinance in April, it was merely to modernize what had existed for some time, Police Chief Robert Howell said': '1',
         'Largo passed a curfew in February but granted a grace period to inform kids about it':'1', 'Moore and Oklahoma City passed curfew ordinances last month':'1', 'A curfew law was passed about a month ago in Pocatello':'1', 'The state judges decision probably invalidates most of the curfew laws on the books in at least two dozen Washington cities, including La Center, which enacted a curfew ordinance similar to Bellinghams 15 months ago':'1', 'Edmond City Attorney Steve Murdock said the current curfew ordinance was enacted in 1974':'1', 'South Charleston City Council enacted a curfew law in December 1993':'1', 'Pocatello and Blackfoot have recently passed similar ordinances':'1', 'Judge Emmet Sullivan said the District of Columbia Council had passed the curfew without credible statistical evidence that youths commit more crimes or become victims more often during the hours of the curfew: 11 P.M':'1',
         'Lakeland passed an ordinance last month requiring youths younger than 16 to be home by 11 p.m':'1', 'in November 1994, Pawtucket enacted a curfew that served as a model for the one the council approved last night':'1','The DeSoto County Commission unanimously passed an ordinance limiting the times juveniles are allowed to be on the streets, aligning itself with the city curfew enacted last month':'1', 'Largo passed a curfew in February modeled after Pinellas Parks':'1', 'Tampa passed a curfew in January, prohibiting people younger than 17 from being on the streets from 11 p.m':'1', 'Tampa has had a curfew for children under 17 since January 1994':'1', 'Largo adopted a similar ordinance in 1998':'1', 'Lakeland passed an ordinance last month requiring youths younger than 16 to be home by 11 p.m': '1', 'The DeSoto County Commission unanimously passed an ordinance limiting the times juveniles are allowed to be on the streets, aligning itself with the city curfew enacted last month':'1',
         'South Charleston City Council enacted a curfew law in December 1993':'1', 'the council passed a juvenile curfew ordinance modeled on one used in Dallas':'1', 'Joseph County Council gave unanimous approval Tuesday to a curfew for juveniles in the county areas outside South Bend and Mishawaka':'1', 'After A Rash Of Car Thefts In Lake Wales Declared An Emergency And Enacted The Countys First Youth Curfew':'1', 'Winter Haven Enacted A Curfew In October A Month After Frostproof Passed Such An Ordinance':'1', 'The Youth Protection Ordinance Passed By City Council At The End Of Requires People Under The Age Of To Be Off City Streets By Pm On Weekdays And Midnight On The Weekends': '1', 'Illinois Passed Its First Curfew Ordinance In July':'1', 'The Pinellas Park Ordinance Was Passed In And Went Into Effect That June':'1', 'The City Passed The Curfew Law In December In An Attempt Not Only To Protect But To Curb Juvenile He Said':'1', 'Metivier Has Signed The Juvenile Curfew Ordinance Passed By The City Council Last Week':'1',
         'This The New Orleans Police Department Reinstituted Walking Reassigned Police Officers From Desk Jobs To Street And Enacted One Of The Countrys Toughest Juvenile':'1', 'The State Of Florida Recently Passed A Law That Severely Limits The Time Youngsters Can Drive At Night':'1', 'The City Passed The Curfew Law In December In An Attempt To Curb Juvenile Crime And Protect Youngsters':'1', 'County Officials Last Year Enacted A Forcing Teens Under Off The Street By Pm On Weeknights And Midnight On Weekends':'1', 'The City Passed The Curfew Law In December In An Attempt Not Only To Protect But To Curb Juvenile Crime':'1', 'Passed An Ordinance Last Month Requiring Youths Younger Than To Be Home By Pm On Weeknights And Midnight On Weekends':'1', 'The Desoto County Commission Unanimously Passed An Ordinance Limiting The Times Juveniles Are Allowed To Be On The Aligning Itself With The City Curfew Enacted Last Month':'1', 'He Enacted One Of The Toughest Juvenile Curfews In The A Move Credited A Year Later With Cutting Juvenile Crime By Percent':'1',
         'After A Rash Of Car Thefts In Lake Wales Declared An Emergency And Enacted The Countys First Youth Curfew':'1', 'Winter Haven Enacted A Curfew In October A Month After Frostproof Passed Such An Ordinance':'1', 'A Curfew Enacted In Two Years Ago Decreased Crimes Involving Young People To Percent During The Hours Of The She Said':'1', 'The Curfew Enacted There In February':'1', 'Exceptions To Curfews The Bill Enacted Today Lets Municipalities Make It Unlawful For Anyone Under To Be Any Public Street Or In A Public Place Between The Hours Of':'1', 'The Proposed Curfew Is Similar To One Recently Enacted In':'1','The Lake Wales Law Enacted In Forbids Youths Age And Under From Hanging Around Outside After Pm Sunday Through And After Midnight On Friday And Saturday':'1', 'Passed A Curfew In February Modeled After Pinellas Parks':'1', 'The Measure Passed June By The City Council Imposes A Curfew Of Pm To Am Sunday Through Thursday And Midnight To Am Friday And Saturday For Unaccompanied Youngsters Under Years Of Age':'1', 'Riverdale Passed A Juvenile Curfew April And Has Issued Warnings Under The Which Provides A Maximum Fine Against Parents Or Owners Of Facilities Where Children Congregate After Pm During The':'1'}
final_sample = {}
for key,value in sample.items():
    if key not in final_sample.keys():
        final_sample[key] = value


# In[47]:


sample_zero = {'The proposed curfew would mirror one going into effect at 12:01 Saturday morning in Lakeland':'0', 'In Tampa, Mayor Sandy Freedman vetoed a juvenile curfew for Ybor City after the ACLU promised to sue if the curfew were enacted':'0','The Dallas curfew is a case in point':'0', 'The Denver curfew program enjoys a collaborative partnership with 234 community programs to which children and their families are diverted':'0', 'In support of the curfew ordinance, the Jacksonville Police Department, the Duval County Parks, Recreation, and Entertainment Department, and the Duval County School Board provide a range of community-based delinquency prevention programs':'0', '"We oppose the law because it violates the constitutional rights of teenagers," says Arthur Spitzer, legal director of the ACLUs Washington chapter':'0', 'The Tampa ordinance gave parents more leeway, he wrote, by allowing children to break curfew to run general errands for their parents, for example':'0', 
                'The highest courts in Iowa, Hawaii and Washington state have found curfews unconstitutional, while other courts have upheld curfews in Charlottesville, Va., and Dallas':'0', 'A 17-year-old South Bend teen received a ticket about 15 minutes past curfew on March 12 at the intersection of Cleveland Road and Riverside Drive':'0', 'The West Virginia justices - with Starcher dissenting - acknowledged that an ordinance in Charleston infringed on some civil liberties, but said the impact was not severe enough to be unconstitutional':'0', 'The thing is, Burke initially opposed the curfew in Largo':'0', 'In Dallas in 1995, the first full year of the curfew, the police picked up 4,000 young people, of whom 2,500 were repeat offenders who were given citations ordering them to court':'0',
              'He also ruled the city of Indianapolis policy of subjecting minors who break curfew to drug and alcohol tests violates the Fourth Amendment':'0', '"Juvenile crime has increased at a greater rate than adult crime in Dubuque in the 90s," Mauss said':'0','Pasco Sheriff Lee Cannon expects to reintroduce a proposed juvenile curfew to Pasco commissioners next week':'0', 'San Diego will continue to be one of the nations safest large cities. New York City had the lowest overall crime rate among big cities':'0',
              'Taylor thinks the curfew has been effective':'0','Kenton Tarver, East Lake High Editor: The one crime this curfew law wont stop is the robbery of memories, good times and friendship that, according to Tampa City Council, must end at 10 p.m':'0', 'Pedestrian killed on Providence Road BRANDON - A pedestrian died after being hit by a car Tuesday evening on Providence Road south of Brandon Boulevard':'0', '"The word has gotten out to the kids, stay off the streets after hours." Since the curfew took effect, Lakeland police have reported a 41 percent overall reduction in juvenile crime':'0', 'Strauss , that the exemptions under the Dallas ordinance, which permitted juveniles to exercise their fundamental rights and remain in public, demonstrated that the ordinance was narrowly tailored to meet the citys legitimate objectives':'0',
              'Rice acknowledged that Newark already has a curfew, but he said that it has proved unenforceable, largely, he maintained, because parents were not held responsible for their childrens violations':'0', 'A curfew in Largo also could be affected, but officials said it was too early to tell':'0', 'So, many of those who want to race, or just pose as if they were racers, go to illegal street events like these, or to similar industrial areas in Ontario, Gardena and Huntington Beach':'0', 'Tampa passed a similar curfew two years ago, but for the most part it has gone unenforced because police and city attorneys believed it would not stand up under legal challenges':'0', 'He expects something similar to happen if Largo enacts its curfew':'0',
              'Oklahoma City also has a curfew rule, patterned after a similar ordinance in Dallas':'0', 'Fourteen years later, in 1989, Simbi Waters challenged a juvenile curfew ordinance in the District of Columbia on the grounds that it violated her first, fourth, and fifth amendment rights':'0', 'With the Powder Springs City Councils adoption this week of a teen curfew':'0', 'In Tampa, Mayor Sandy Freedman vetoed a juvenile curfew for Ybor City after the ACLU promised to sue if the curfew were enacted':'0', 'Annie Brown Kennedy of Winston-Salem, N.C., the first black woman to serve in the North Carolina Legislature, announced that shell retire when her term ends in 1994':'0', '5841 Roswell Road: The 16-year-old girl had passed out in the Sandy Springs bar and an ambulance was called':'0', 'Would Go To Court To Have Any Ordinances Overturned If Any Are Passed And Any Parents Or Juveniles Are Charged As A Result':'0', 'Between January And The Of An Additional Of These Cities Enacted Juvenile Curfew Bringing The Total Of Those With Curfew Laws To':'0',
              'Minnesota Which Held That A Parental Notification Requirement Of The States Abortion Statute Passed Constitutional Muster Because States Have':'0', 'A Review Of The Citys Original Curfew Enacted In Found It Ambiguous And Unenforceable':'0', 'In Addition To The Curfew Enforcement Has Strengthened Its Commitment To Crime Prevention And Reduction Through Community Newly Enacted Weapon And Programs In Elementary And Junior High Schools':'0', 'Currys Motion That The Task Force Be Formed Was Unanimously Passed By All Three Trustees':'0', 'In Other The Council Will Consider Condemnation Of Property Owned By New Resident Dino Levi At Nelson St':'0', 'Earlier This Week Frostproof Reaffirmed Its Curfew A Necessary Step Because The Law Was Passed As An Emergency Measure':'0', 'It Is Very Timely In The Wake Of All This Youth Other Council Did Not Share Mr':'0', 'The Chapter Plans To Challenge The Curfew Enacted There In February':'0',
              'On Monday He Became The First Town Council President In Anyones Memory To Hold Onto The Top Post':'0', 'If The Result Holds The Charter Will Bring Its First Town Will Increase Council Terms From Two To Four Years And Will Allow Voters To Recall A Council Member Midterm':'0', 'Though Were Not Were Still Citizens Of The United Even Council Member Jay Who Proposed The Ended Up Voting Against It':'0', 'Two Speakers Presented Petitions To The Council They Said Contained Signatures Opposing A Curfew':'0', 'Then About Minutes Passed And A Nice Young Couple Stopped And Offered Assistance':'0', 'The City Council Defeated A More Restrictive Curfew Earlier This But Vieira Predicts The Council Will Embrace His Proposal Because The Public Is Fed Up With And Because Police Chief Richard E':'0',
              'The Council Will Vote On The Ordinance For The First Time May':'0', 'In Just The Last Five Over Of These Major Cities Either Enacted A New Curfew Or Revised An Existing One Including Such Liberal Cities As San And':'0', 'The Neighborhood Is Represented By Council Members Ted Wright And Gail Both Who Told Residents That The Council Might Decide The Issue At The Next Council July':'0', 'A New Dna Part Of A Package Recently Enacted By The May Be Illegal On Technical Says Bobby Timmons Of The State Sheriffs Assn':'0','In More Than Of The Largest Cities In The United States Have Passed Curfew Legislation In The Past Five Bringing To More Than The Number With Such Laws In Effect':'0', 'Last Council Member Rudy Fernandez Embraced A Bill By State Sen':'0', 'Juvenile Council Members Questioned Whether The City Should Enact An Ordinance It Might Not Be Able To Enforce':'0', 'Officials In Winter Fort Frostproof And Lake Wales Already Have Passed Similar Ordinances':'0'}
final_sample_zero = {}
for key,value in sample_zero.items():
    if key not in final_sample_zero.keys():
        final_sample_zero[key] = value


# In[79]:


print(len(sample), len(sample_zero))


# In[80]:


final_sample.update(final_sample_zero)


# In[81]:


raw_feature = []
labels = []
feature_set = []
for key, value in final_sample.items():
    raw_feature.append(key.title())
    labels.append(value)


# In[82]:


print(len(raw_feature), len(labels))


# In[83]:


x_train, x_test, y_train, y_test = tts(raw_feature, labels, test_size= 0.2)


# In[87]:


newpipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
                    ('chi', SelectKBest(chi2, k=100)),
                    ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


# In[88]:


model = newpipeline.fit(x_train, y_train)


# In[89]:


print(str(model.score(x_test, y_test)))


# In[90]:


hits = []
for sents in corpus:
    list_sents = [sents]
    result = model.predict(list_sents)
    if result[0] == '1':
        hits.append(sents)
        print(sents)


# In[93]:


hits = []
for sents in table['Sample']:
    for sentence in sents:
        list_sents = [sentence]
        result = model.predict(list_sents)
        if result[0] == '1':
            hits.append(sentence)
            print(sentence)
            print("-"*50)
        


# In[94]:


final_indicated_city = []
ordered_hits = []
data_date = []
final_stated_city = []
for i in range(len(table)):
    matcher = []
    for x in table['Sample'][i]:
        if x in hits:
            ordered_hits.append(x)
            final_indicated_city.append(table['Indicated City'][i])
            data_date.append(table['Date'][i])
            final_stated_city.append(table['Stated City'][i])
            print(x,table['Date'][i],table['Indicated City'][i])


# In[95]:


ss = pd.DataFrame({'City':final_indicated_city, 'Sentence':ordered_hits, 'Date':data_date, 'Stated City': final_stated_city})
ss


# In[96]:


for i in range(len(table)):
    sentence_data = ss['Sentence'][i]
    city_data = ss['City'][i]
    stated_city_data = ss['Stated City'][i]
    date_data = ss['Date'][i]
    print("Cited City", city_data, "||||", "Stated City", stated_city_data)
    print('Text: ', sentence_data)
    print(date_data)
    print('*'*100)


# In[ ]:




