import numpy as np
wordsList = np.load('./imdb/wordList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() # Originally loaded as numpy array
# code as utf-8
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('./imdb/wordVectors.npy')
print('Loaded the word vector')

print(len(wordsList))
print(wordVectors.shape)

'''
baseballIndex = wordsList.index('baseball')
wordVectors[baseballIndex]
'''

# embed a sentence to analyse
maxSeqLength = 10 # max length of a sentence
numDimensions = 300 # dim of word vectors
firstSentence = np.zeros((maxSeqLength),dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
# firstSentence[8] and firstSentence[9] = 0
print(firstSentence.shape)

from os import listdir
from os.path import isfile,join
positiveFiles = ['./imdb/positiveReviews/'+ f for f in listdir('./imdb/positiveReviews/')if isfile(join('./imdb/positiveReviews/',f))]
negativeFiles = ['./imdb/negativeReviews/'+ f for f in listdir('./imdb/negativeReviews/')if isfile(join('./imdb/negativeReviews/',f))]
numWords = []
for pf in positiveFiles:
    with open(pf,"r",encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Positive files finished')

for nf in negativeFiles:
    with open(nf,"r",encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')

numFiles = len(numWords)
print('The total number of files is',numFiles)
print('The total number of words in the files is',sum(numWords))
print('The average number of words in the files is',sum(numWords)/len(numWords))

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='/home/wumg/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf')

plt.hist(numWords,50)
plt.xlabel('length of seq',fontproperties=myfont)
plt.ylabel('frequency',fontproperties=myfont)
plt.axis([0,1200,0,8000])
plt.show()

maxSeqLength = 250
fname = positiveFiles[3] #can use any valid index
with open(fname) as f:
    for lines in f:
        print(lines)
        exit()

# delete signs, only leave number and words
import re
strip_special_chars = re.compile("[A-Za-z0-9 ]+")
def cleanSentence(string):
    string = string.lower().replace("<br /"," ")
    return re.sub(strip_special_chars,"",string.lower())

firstFile = np.zeros((maxSeqLength),dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentence(line)
    split = cleanedLine.split()
    for word in split:
        try:
            firstFile[indexCounter]=wordsList.index(word)
        except ValueError:
            firstFile[indexCounter] = 399999 #vector for unknown words
        indexCounter = indexCounter+1

ids = np.load('./imdb/idsMatrix.npy')

# define functions
from random import randint
def getTrainBatch():
    labels=[]
    arr = np.zeros([batchSize,maxSeqLength])
    for i in range(batchSize):
        if(i%2==0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr,labels

def getTestBatch():
    labels=[]
    arr = np.zeros([batchSize,maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if(num<=12499):
            labels.append([1, 0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr,labels

# define hyper parameter
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 20000

import tensorflow as tf
tf.reset_default_graph()
labels = tf.placeholder(tf.float32,[batchSize,numClasses])
input_data = tf.placeholder(tf.int32,[batchSize,maxSeqLength])
data = tf.nn.embedding_lookup(wordVectors,input_data)
#construct a RNN model
lstmCell = tf.contrib.rnn.BasicLSTMCELL(lstmUnits)
lstmCell = tf.contrib.DropoutWrapper(cell=lstmCell,output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits,numClasses]))
bias = tf.Variable(tf.constant(0.1,shape=[numClasses]))
value = tf.transpose(value,[1,0,2])
last = tf.grapher(value,int(value.get_shape()[0])-1)
prediction = (tf.matmul(last,weight)+bias)

correctPred = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred,tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime
sess = tf.Session()
tf.summary.scalar('Loss',loss)
tf.summary.scalar('Accuracy',accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/"+datetime.datetime.now().strftime(("%Y%m%d-%H%M%S")+"/")
writer = tf.summary.FileWriter(logdir,sess.graph)

#train the model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
with tf.device('/gpu:0'):
    for i in range(iterations):
        # get data in next batch
        nextBatch,nextBatchLabels = getTrainBatch()
        sess.run(optimizer,{input_data:nextBatch,labels:nextBatchLabels})

        if(i%50==0):
            summary = sess.run(merged,{input_data:nextBatch,labels:nextBatchLabels})
            writer.add_summary(summary,i)

        if (i%1000==0 and i!=0):
            save_path = saver.save(sess,"models/pretrained_lstm.ckpt",global_step=1)
            print("saved to %s"%save_path)

    writer.close()


# save restore model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint('./models'))

iterations = 10
for i in range(iterations):
    nextBatch,nextBatchLabels=getTestBatch()
    print("accuracy for this batch:",(sess.run(accuracy,{input_data:nextBatch,labels:nextBatchLabels}))*100)





