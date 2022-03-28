from pprint import pprint
from Parser import Parser
import util
from tfidf import *
import glob, os
import math
import numpy as np
from numpy.linalg import norm
from textblob import TextBlob as tb

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None
    
    def __init__(self, documents=[], vectorMode = 'tf'):
        self.documentVectors=[]
        self.parser = Parser()
        self.BlobList = self.getBlobList(documents)
        self.vectorMode = vectorMode
        if(len(documents)>0):
            self.build(documents)
    
    def getBlobList(self, documents): 
        bloblist = []
        for doc in documents:
            wordList = self.parser.tokenise(doc)
            wordList = self.parser.removeStopWords(wordList)
            bloblist.append(tb(" ".join(wordList)))
        return bloblist
            
    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document, self.vectorMode) for document in documents]

        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)
        
        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString, mode):
        """ @pre: unique(vectorIndex) """
        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        tbString = tb(" ".join(wordList))
        #print(wordList)
        if mode == 'tf':
            for word in list(set(wordList)):
                vector[self.vectorKeywordIndex[word]] = tf(word, tbString) #Use simple Term Count Model
            return vector 
        
        if mode == 'tf-idf':
            #print('bloblist:', self.BlobList)
            for word in list(set(wordList)):
                #print('word',word)
                vector[self.vectorKeywordIndex[word]] =  tfidf(word, tbString , self.BlobList) 
                #print('word:',word, 'idf:', idf(word, self.BlobList),  )
                #print('word:', word, 'tf:', tf(word, tbString))
            return vector
        
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        #print(termList)
        #print(self.vectorMode)
        query = self.makeVector(" ".join(termList), self.vectorMode)
        return query

    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    def search(self,searchList, mode = 'cos'):
        """ search for documents that match based on a list of terms """
        #print(searchList)
        if type(searchList[0]) == str:
            queryVector = self.buildQueryVector(searchList)
            #print(queryVector)
        else:
            queryVector = searchList
        
        if mode == 'cos':
            ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
            return ratings
        if mode == 'eucli':
            ratings = [util.Euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
            return ratings

    def printresult(self,searchlist, files,n, mode='cos'):
        scoreList = self.search(searchlist, mode = mode)
        if self.vectorMode == 'tf' and mode == 'cos':
            print('Term Frequency (TF) Weighting + Cosine Similarity\n')
        elif self.vectorMode == 'tf' and mode == 'eucli':
            print('Term Frequency (TF) Weighting + Euclidean Distance\n')
        elif self.vectorMode == 'tf-idf' and mode == 'cos':
            print('TF-IDF Weighting + Cosine Similarity\n')
        elif self.vectorMode == 'tf-idf' and mode == 'eucli':
            print('TF-IDF Weighting + Euclidean Distance\n')
        print( 'NewsID' ,'         ','score')
        print('----------','     ','--------')
        for i in np.flip(np.argsort(scoreList))[:n]:
            print(files[i],'     ' ,scoreList[i])
        print('')
        return np.flip(np.argsort(scoreList))[:n]
            
    def getFeedbackVector(self, searchList, top10results ,mode = 'cos'):
        queryVector = self.buildQueryVector(searchList)
        feedbackVector = self.buildQueryVector(top10results)
        queryArray = np.array(queryVector)
        feedbackArray = np.array(feedbackVector)
        newQueryVector = list(queryArray + 0.5 * feedbackArray)
        return newQueryVector

if __name__ == '__main__':
    documents = []
    files = []
    for file in os.listdir("./EnglishNews/EnglishNews"):
        if file.endswith(".txt"):
            filename = os.path.join("./EnglishNews/EnglishNews", file)
            files.append(file[:-4])
            with open(filename, encoding="utf-8") as f:
                lines = f.readlines()
                doc = ' '.join(lines)
                doc1 = doc.replace("\n", "")
                documents.append(doc1)
                
    query = ["Trump Biden Taiwan China"]
    '''
    vectorSpace_tf = VectorSpace(documents, 'tf')
    
    rank_cos = vectorSpace_tf.printresult(query, files,10,'cos')  #60% 前30: 100% 
    rank_euc = vectorSpace_tf.printresult(query, files, 10,'eucli')  #50% 前30: 80%
    '''
    vectorSpace_tfidf = VectorSpace(documents,'tf-idf')
    
    rank_cos_tfidf = vectorSpace_tfidf.printresult(query,files, 10,'cos')#70% #前30個100%
    rank_euc_tfidf = vectorSpace_tfidf.printresult(query,files, 10,'eucli')#40% #前30個40%
    

    # Q2
    print('question 2\n')
    feedBack = []
    for i in rank_cos_tfidf[:10]:
        feedBack.append(documents[i])
        feedback = [" ".join(feedBack)]
    newquery = vectorSpace_tfidf.getFeedbackVector(query, feedback, 'cos')
    
    vectorSpace_tfidf.printresult(newquery,files, 10,'cos')
    