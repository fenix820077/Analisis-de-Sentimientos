# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:58:56 2020

@author: Santy Su
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

tweets=pd.read_csv("C:/Users/Usuario/Desktop/CASO PRACTICO/CASO PRACTICO/TAREA 1.1/tweets.csv",encoding="latin-1")[0:3000]
tweets.info()

#FUNCION PARA LIMPIAR LOS TWEETS

def cleanTxt(text):
 text = re.sub('@[A-Za-z0–9]+', '', text) #REMUEVEN @MENCIONES
 text = re.sub('#', '', text) # REMUEVEN '#' hash tag
 text = re.sub('RT[\s]+', '', text) # REMUEVEN RT
 text = re.sub('https?:\/\/\S+', '', text) # REMUEVEN hyperlink
 
 return text


# LIMPIAMOS LOS TWEETS

tweets['Tweet_depurados'] = tweets['Tweet'].apply(cleanTxt)
# Show the cleaned tweets
tweets

# CREA UNA FUNCIÓN PARA getsubjetivity 


def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# CREA UNA FUNCIÓN PARA to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

# CREA dos nuevas columnas 'Subjectivity' & 'Polarity'
tweets['Subjectivity'] = tweets['Tweet'].apply(getSubjectivity)
tweets['Polarity'] =tweets['Tweet'].apply(getPolarity)

# INDICA O IMPRIME LAS COLUMNAS 
tweets


#GRAFICAMOS DE ACUERDO A LAS PALABRAS MAS MENCIONADAS
allWords = ' '.join([twts for twts in tweets['Tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)


plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


####################################


# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'
tweets['Analysis'] = tweets['Polarity'].apply(getAnalysis)
# Show the dataframe
tweets

# Plotting 
plt.figure(figsize=(8,6)) 
for i in range(0, df.shape[0]):
  plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue') 
# plt.scatter(x,y,color)   
plt.title('Sentiment Analysis') 
plt.xlabel('Polarity') 
plt.ylabel('Subjectivity') 
plt.show()

# Print the percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweet']
ptweets

round( (ptweets.shape[0] / df.shape[0]) * 100 , 1)

# Print the percentage of negative tweets
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweet']
ntweets

round( (ntweets.shape[0] / df.shape[0]) * 100, 1)

df['Analysis'].value_counts()

# Plotting and visualizing the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()
