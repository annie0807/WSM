from pprint import pprint
from Parser import Parser
import util
from tfidf import *
import glob, os
import math
import numpy as np
import jieba
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
    vectorKeywordIndex = []


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

    def is_chinese(self, string):
        for ch in string:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
    
    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        documents_en = []
        documents_ch = []
        #Mapped documents into a single word string	
        for doc in documentList:
            if self.is_chinese(doc) == False:
                documents_en.append(doc)
            else:
                documents_ch.append(doc)
        
        #english part
        vocabularyString= " ".join(documents_en)
        vocabularyList = self.parser.tokenise(vocabularyString) #斷詞
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList) # 去掉stop words
        
        #chinese part
        vocabularyString_ch = " ".join(documents_ch)
        vocabularyList_ch = jieba.lcut_for_search(vocabularyString_ch) #斷詞
        vocabularyList_ch = self.parser.removeChiStopWords(vocabularyList_ch)

        #mix
        vocabularyList.extend(vocabularyList_ch)
        
        uniqueVocabularyList = util.removeDuplicates(vocabularyList) # 去掉重複詞
        #print('去掉重複詞', uniqueVocabularyList)
        
        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        #print(vectorIndex)
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString, mode):
        """ @pre: unique(vectorIndex) """
        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        if self.is_chinese(wordString) == False: #英文文件
            wordList = self.parser.tokenise(wordString)
            wordList = self.parser.removeStopWords(wordList)
            tbString = tb(" ".join(wordList))
        else:                                    #中文文件
            wordList = jieba.lcut_for_search(wordString) #斷詞
            wordList = self.parser.removeChiStopWords(wordList)
            tbString = tb(" ".join(wordList))
        if mode == 'tf':
            for word in list(set(wordList)):
                vector[self.vectorKeywordIndex[word]] = tf(word, tbString) #Use simple Term Count Model
            return vector 

        if mode == 'tf-idf':
            #print('bloblist:', self.BlobList)
            for word in list(set(wordList)):
                vector[self.vectorKeywordIndex[word]] =  tfidf(word, tbString , self.BlobList) 
            return vector
               
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList), self.vectorMode)
        return query

    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    def search(self,searchList, mode = 'cos'):
        """ search for documents that match based on a list of terms """
        if type(searchList[0]) == str:
            queryVector = self.buildQueryVector(searchList)
        else:
            queryVector = searchList
        
        if mode == 'cos':
            ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
            return ratings
        if mode == 'eucli':
            ratings = [util.Euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
            return ratings

    def printresult(self,searchlist, files,n,mode='cos'):
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
        return np.flip(np.argsort(scoreList))[:n]
            
    def getFeedbackVector(self, searchList, top10results ,mode = 'cos'):
        queryVector = self.buildQueryVector(searchList)
        feedbackVector = self.buildQueryVector(top10results)
        queryArray = np.array(queryVector)
        feedbackArray = np.array(feedbackVector)
        newQueryVector = list(queryArray + 0.5 * feedbackArray)
        return newQueryVector

if __name__ == '__main__':
    documents_chi = []
    files_chi = []
    for file in os.listdir("./News/News"):
        if file.endswith(".txt"):
            filename_chi = os.path.join("./News/News", file)
            files_chi.append(file[:-4])
            with open(filename_chi, encoding="utf-8") as f:
                lines = f.readlines()
                doc = ' '.join(lines)
                doc1 = doc.replace("\n", "")
                documents_chi.append(doc1)

    query = ["烏克蘭 大選"]

    vectorSpace_tf = VectorSpace(documents_chi, 'tf')
    vectorSpace_tf.printresult(query, files_chi,10,mode='cos')

    vectorSpace_tfidf = VectorSpace(documents_chi, 'tf-idf')
    vectorSpace_tfidf.printresult(query, files_chi,10,mode='cos') #前30  90%