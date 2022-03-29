#http://tartarus.org/~martin/PorterStemmer/python.txt
from PorterStemmer import PorterStemmer

class Parser:

	#A processor for removing the commoner morphological and inflexional endings from words in English
	stemmer=None

	stopwords=[]
	ChiStop=[]
	def __init__(self,):
		self.stemmer = PorterStemmer()

		#English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
		self.stopwords = open('english.stop', 'r').read().split()
		with open('ChineseStopwords.txt', encoding="utf-8") as f:
			lines =f.readlines()
			stop = ' '.join(lines)
			self.ChiStop = stop.replace("\n", "").split()


	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		string = string.replace(".","")
		string = string.replace("\s+"," ")
		string = string.lower()
		return string
	

	def removeStopWords(self,list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]

	def tokenise(self, string):
		""" break string up into tokens and stem words """
		string = self.clean(string)
		words = string.split(" ")
		
		return [self.stemmer.stem(word,0,len(word)-1) for word in words]
    
	def removeChiStopWords(self, list):
		return [word for word in list if word not in self.ChiStop ]


