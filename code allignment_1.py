import json
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import os
from collections import OrderedDict

df = pd.read_csv("C:/Users/KPATNAIk/Desktop/Topic_Modelling1/one_trust/train.csv",encoding= "latin1")
df["sentence"] = """ "sentence" """ + ":" + '"' +df["Text"]+ '"'+","
df["tags"] = """ "tags" """+ ":" + "{"+'"'+df["Labels"]+'"'+":"+'"'+ df["PII"]+'"'+"}"
df["final_json"] = "{"+df["sentence"]+df["tags"]+"}"
df = df["final_json"]


user_inputk = df.to_dict()
user_inputk1 = list(OrderedDict(sorted(user_inputk.items())).values())
       
f = open ('train.json', "r") 
user_input = json.loads(f.read()) 

for i in range(len(user_input)):        
    user_input_1 = user_input[i]
    wanted_keys = ['sentence']
    wanted_keys1 = ['tags']
    sentence = {k: user_input_1[k] for k in set(wanted_keys) & set(user_input_1.keys())}
    sentence = list( sentence.values() )[0]            
#    sentence = sentence.lower()  
    article = sentence[:]
    def find_match(sentence,df):
        for i in range(df.shape[0]):
            if sentence.find(df['rpl'][i]) !=-1:
                sentence = sentence[:sentence.find(df['rpl'][i])] +  df['rpl1'][i] +  sentence[sentence.find(df['rpl'][i])+ len(df['rpl'][i]):]
        return sentence
                       
    tags = {k: user_input_1[k] for k in set(wanted_keys1) & set(user_input_1.keys())}
    tags = list( tags.values() )[0]
    tags = {k:str(v) for k, v in tags.items()}
#    def lower_dict(d):
#        new_dict = dict((k, v.lower()) for k, v in d.items())
#        return new_dict
#    tags = lower_dict(tags)
    new_list = [] 
    for key, value in tags.items():
        new_list.append([key, value])
    ui1 = pd.DataFrame(new_list)
    ui1.columns = ['action','sentence']
    ui1["sentence1"] = ""
                    
    uik = ui1[ui1["sentence1"]=="Found"]
    uik["sentence"] = uik["sentence"].astype(str).str[:-6] #Strip time zone        
    uik["sentence"] = pd.to_datetime(uik["sentence"], errors='coerce')
    uik["sentence"] = uik["sentence"].dt.strftime('%Y-%m-%d')            
    uik = uik.query('sentence != "NaT"')
    uik = uik.drop(['sentence1'], axis=1)
    ui1 = ui1.drop(['sentence1'], axis=1)
    ui1 = ui1.append(uik, ignore_index=True) 
                    
        
    k = ui1.apply(lambda row: nltk.word_tokenize(row['sentence']), axis=1)
    k = pd.DataFrame(k)
    k.columns = ["sentence"]
    new = k.sentence.apply(pd.Series)
    new["action"]=ui1["action"]
    df_new = pd.DataFrame()
    for label, content in new.items():
        df_new1 = pd.DataFrame()
        df_new1[0] = new["action"]
        df_new1[1] = new[label]
        df_new = df_new.append(df_new1, ignore_index=True)
        df_new = df_new[df_new[0] != df_new[1]]
        df_new = df_new.dropna()
    lst_ip1 = word_tokenize(sentence)
    lst_ip3 = pd.DataFrame(lst_ip1)
    lst_ip3.columns = ['sentence']
    df_new.columns = ['action','sentence']
    #################################################join
    result = pd.merge(lst_ip3,
                     df_new,
                     on='sentence', 
                     how='left')             
    
    result['action'] = result['action'].fillna('o')
    result['key'] = (result['sentence'] != result['sentence'].shift(1)).astype(int).cumsum()
    result =result.groupby(['key', 'sentence'])['action'].apply('#$#'.join).to_frame()
    result = result.reset_index()
    result['sentence'] = result['sentence'].map(str) + " " + result["action"]
    user_input3 = result['sentence']
    user_input3.to_csv('user_input3.tsv',header=False, index=False)
    user_input3 = pd.read_csv('user_input3.tsv', sep='\t',header = None)
    exists = os.path.isfile('dummy-corpus1.tsv')
    exists1 = os.path.isfile('dummy-corpus2.tsv')
    if exists and not exists1:
        pa1 = pd.read_csv('dummy-corpus1.tsv', sep='\t',header = None)
        pa2 = pa1.append(user_input3,ignore_index=True)
        pa2 = pa2.append(["**** o"])
    elif exists1 and exists:
        pa1 = pd.read_csv('dummy-corpus2.tsv', sep='\t',header = None)
        pa2 = pa1.append(user_input3,ignore_index=True)
        pa2 = pa2.append(["**** o"])  
    else:
        pa2 = user_input3
        pa2 = pa2.append(["**** o"])
        
    pa2.to_csv('dummy-corpus1.tsv',header=False, index=False)