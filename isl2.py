import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import sys
import argparse
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tree import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk

inputString = " "
import os
java_path = "C:\\Program Files\\Java\\jdk-13.0.1\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_PARSER'] = r"C:\\Users\\shree\\Anaconda3\\Lib\\site-packages\\jars"
os.environ['STANFORD_MODELS'] = r"C:\\Users\\shree\\Anaconda3\\Lib\\site-packages\\jars"
os.environ['CLASSPATH']=r"D:/stanford-parser-full-2018-02-27"

#import selecting
# obtain audio from the microphone
def func():
        r = sr.Recognizer()
        isl_gif=['all the best', 'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
                 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call an ambulance', 'please call me later',
                'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                'what are you doing', 'what is the problem', 'what is todays date', 'what is your age', 'what is your father do', 'what is your job',
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
'voice', 'wednesday', 'weight']
        
        
        arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
        's','t','u','v','w','x','y','z']
        with sr.Microphone() as source:

                r.adjust_for_ambient_noise(source) 
                i=0
                while True:
                        print('Say something')
                        audio = r.listen(source)

                                                        # recognize speech using Sphinx
                        try:
                                a=r.recognize_google(audio)
                                print("you said " + a.lower())
                                inputString =a.lower()
                                parser=StanfordParser(model_path='D:/stanford-parser-full-2018-02-27/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
                                o=parser.parse(inputString.split())
                                #print(o)
                                englishtree=[tree for tree in parser.parse(inputString.split())]
                                parsetree=englishtree[0]
                                dict={}
                                # "***********subtrees**********"
                                parenttree= ParentedTree.convert(parsetree)
                                for sub in parenttree.subtrees():
                                        dict[sub.treeposition()]=0
                                        #"----------------------------------------------"
                                isltree=Tree('ROOT',[])
                                i=0
                                for sub in parenttree.subtrees():
                                        if(sub.label()=="NP" and dict[sub.treeposition()]==0 and dict[sub.parent().treeposition()]==0):
                                                dict[sub.treeposition()]=1
                                                isltree.insert(i,sub)
                                                i=i+1
                                        if(sub.label()=="VP" or sub.label()=="PRP"):
                                                for sub2 in sub.subtrees():
                                                        if((sub2.label()=="NP" or sub2.label()=='PRP')and dict[sub2.treeposition()]==0 and dict[sub2.parent().treeposition()]==0):
                                                                dict[sub2.treeposition()]=1
                                                                isltree.insert(i,sub2)
                                                                i=i+1
                                for sub in parenttree.subtrees():
                                        for sub2 in sub.subtrees():
                                                # print sub2
                                                # print len(sub2.leaves())
                                                # print dict[sub2.treeposition()]
                                                if(len(sub2.leaves())==1 and dict[sub2.treeposition()]==0 and dict[sub2.parent().treeposition()]==0):
                                                        dict[sub2.treeposition()]=1
                                                        isltree.insert(i,sub2)
                                                        i=i+1
                                parsed_sent=isltree.leaves()
                                words=parsed_sent
                                stop_words=set(stopwords.words("english"))
                                # print stop_words
                                lemmatizer = WordNetLemmatizer()
                                ps = PorterStemmer()
                                lemmatized_words=[]
                                for w in parsed_sent:
                                        # w = ps.stem(w)
                                        lemmatized_words.append(lemmatizer.lemmatize(w))
                                islsentence = ""
                                print("According to ISL:")
                                print(lemmatized_words)
                                for w in lemmatized_words:
                                        if w not in stop_words:
                                                islsentence+=w
                                                islsentence+=" "
                                l=islsentence.split(" ")
                                #print(l)
                                t=set(l)
                                l1=list(t)
                                #print(l1)
                                #print("")
                                while("" in l1) : 
                                    l1.remove("")
                                print(l1)
                                str=" "
                                str=str.join(l1)
                                print("Output:")
                                print(str)
                                
                                #print(islsentence) 

                                for c in string.punctuation:
                                    a= a.replace(c,"")
                                    
                                if(str=='done'):
                                        print("oops!Time To say good bye")
                                        break
                                
                                elif(a.lower() in isl_gif):
                                    
                                    class ImageLabel(tk.Label):
                                            """a label that displays images, and plays them if they are gifs"""
                                            def load(self, im):
                                                if isinstance(im, a.lower()):
                                                    im = Image.open(im)
                                                self.loc = 0
                                                self.frames = []

                                                try:
                                                    for i in count(1):
                                                        self.frames.append(ImageTk.PhotoImage(im.copy()))
                                                        im.seek(i)
                                                except EOFError:
                                                    pass

                                                try:
                                                    self.delay = im.info['duration']
                                                except:
                                                    self.delay = 100

                                                if len(self.frames) == 1:
                                                    self.config(image=self.frames[0])
                                                else:
                                                    self.next_frame()

                                            def unload(self):
                                                self.config(image=None)
                                                self.frames = None

                                            def next_frame(self):
                                                if self.frames:
                                                    self.loc += 1
                                                    self.loc %= len(self.frames)
                                                    self.config(image=self.frames[self.loc])
                                                    self.after(self.delay, self.next_frame)

                                    root = tk.Tk()
                                    lbl = ImageLabel(root)
                                    lbl.pack()
                                    lbl.load(r'C:/Users/shree/ISL/ISL_Gifs/{0}.gif'.format(a.lower()))
                                    root.mainloop()
                                else:

                                    for i in range(len(a)):
                                                    #a[i]=a[i].lower()
                                                    if(a[i] in arr):
                                            
                                                            ImageAddress = 'letters/'+islsentence[i]+'.jpg'
                                                            ImageItself = Image.open(ImageAddress)
                                                            ImageNumpyFormat = np.asarray(ImageItself)
                                                            plt.imshow(ImageNumpyFormat)
                                                            plt.draw()
                                                            plt.pause(1) # pause how many seconds
                                                            #plt.close()
                                                    else:
                                                            continue

                        except:
                               print("Could not listen")
                        plt.close()

#func()
while 1:
  image   = "signlang.png"
  msg="Speech To Sign Converter"
  choices = ["Live Voice","All Done!"]
  reply   = buttonbox(msg,image=image,choices=choices)
  if reply ==choices[0]:
        func()
  if reply == choices[1]:
        quit()
 # if reply==choices[1]:
  #     from main import cap
