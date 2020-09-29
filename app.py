from nltk.tag import StanfordNERTagger  
import nltk
import json
from flask import Flask, render_template, request
import pandas as pd
jar = 'C:/Users/KPATNAIk/Desktop/Topic_Modelling1/stanford-ner.jar'
model = 'C:/Users/KPATNAIk/Desktop/Topic_Modelling1/one_trust/corpus-tagging.ser.gz'

app = Flask(__name__)
@app.route('/',methods = ['GET', 'POST'])
def upload_file():
   return render_template('index.html')
@app.route('/data',methods = ['GET', 'POST'])
def data():
    if request.method=="POST":
        f = request.form['csvfile']
        df = pd.read_csv(f,encoding= "latin1")
#        df = pd.read_csv("C:/Users/KPATNAIk/Desktop/Topic_Modelling1/one_trust/test.csv",encoding= "latin1")
        df = df.head(10)
        df['Text1'] = df['Text'] +" "+"****"
        df['Text1'] = "****"+" "+df['Text1']
        ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
        for i, row in df.iterrows():
            words = nltk.word_tokenize(row["Text1"])
            results = ner_tagger.tag(words)
            filter = ['o']
            ls2 = [(x,y) for (x,y) in results if y not in filter] 
            d = {}
            for a, b in ls2:
                d.setdefault(b, []).append(a)
            new_abc = [ [ ' '.join(d.pop(b)), b ] for a, b in ls2 if b in d ]
            if len(new_abc) != 0:
                df.at[i,'Labels'] = new_abc[0][1]
                df.at[i,'PII'] = new_abc[0][0]
            else:
                df.at[i,'Labels'] = "None"
                df.at[i,'PII'] = "None"
        df = df.drop(["Text1"],axis = 1)
        df.to_csv("test1.csv")
        data = df
        return render_template('data.html',data=data.to_html())

if __name__ == '__main__':
    app.run("0.0.0.0",threaded=False)
