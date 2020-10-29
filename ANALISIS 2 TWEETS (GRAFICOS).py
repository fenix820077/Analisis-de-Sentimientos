# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:58:56 2020

@author: Santy Subia
"""

# Import the libraries
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
plt.style.use('fivethirtyeight')

df = pd.read_csv("C:/Users/Usuario/Desktop/CASO PRACTICO/CASO PRACTICO/TAREA 1.1/tweets.csv",encoding="latin-1")[0:3000]
stop_words = list(set(stopwords.words("spanish")))

#FUNCION PARA LIMPIAR LOS TWEETS

def cleanTxt(text):
 text = re.sub('@[A-Za-z0–9]+', '', text) #REMUEVEN @MENCIONES
 text = re.sub('#', '', text) # REMUEVEN '#' hash tag
 text = re.sub('RT[\s]+', '', text) # REMUEVEN RT
 text = re.sub('https?:\/\/\S+', '', text) # REMUEVEN hyperlink
 
 return text


# LIMPIAMOS LOS TWEETS

df['Tweet_depurados'] = df['Tweet'].apply(cleanTxt)

# ENSEÑAMOS EL DF LIMPIO
df

# CREA UNA FUNCIÓN PARA getsubjetivity 

def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# CREA UNA FUNCIÓN PARA to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

# CREA dos nuevas columnas 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['Tweet'].apply(getSubjectivity)
df['Polarity'] =df['Tweet'].apply(getPolarity)

# INDICA O IMPRIME LAS COLUMNAS 
df


#GRAFICAMOS DE ACUERDO A LAS PALABRAS MAS MENCIONADAS
allWords = ' '.join([twts for twts in df['Tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)


plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


####################################

# CREA FUNCIONES POSITIVAS; NUETRALES ;NEGATIVAS
def getAnalysis(score):
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)
# INDICA EL DATAFRAME
df

# GRAFICO  
plt.figure(figsize=(8,6)) 
for i in range(0, df.shape[0]):
  plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Red') 

# plt.scatter(x,y,color)   
plt.title('Sentiment Analysis') 
plt.xlabel('Polarity') 
plt.ylabel('Subjectivity') 
plt.show()

# IMPRIME SOLO LOS COMENTARIOS POSITIVOS
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweet']
ptweets

round( (ptweets.shape[0] / df.shape[0]) * 100 , 1)

# IMPRIME COMENTARIOS NEGATIVOS
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweet']
ntweets

round( (ntweets.shape[0] / df.shape[0]) * 100, 1)

df['Analysis'].value_counts()
########################
print("Polaridad")
print("Maximo valor", df["polaridad"].max())
print("Maximo valor", df["polaridad"].min())
print("Valor medio", df["polaridad"].mean())

sns.distplot(df["polaridad"])
sns.distplot(df["subobj"])

n=1
while n<5:
    df["P "+str(n)]=df["content"].apply(lambda x: TextBlob(x).)
    df["S "+str(n)]=df["content"].apply(lambda x: TextBlob(x)).
    n=n+1
    
df.to_csv("res.csv")
# GRAFICO DE POSITIVOS NEGATIVOS Y NEUTROS
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()



#FIN DEL ANALISI2S TWEETS