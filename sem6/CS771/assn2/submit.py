import numpy as np

# You are not allowed to import any libraries other than numpy

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF PROHIBITED LIBRARIES WILL RESULT IN PENALTIES

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

################################
# Non Editable Region Starting #
################################
def my_fit( words, num_words ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your decision tree model using the word list provided
	# Return the trained model as is -- do not compress it using pickle etc
	# Model packing or compression will cause evaluation failure
	model = Tree( min_leaf_size = 1, max_depth = 15 )
	model.fit( words )
	return model					# Return the trained model

class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words ):
		self.words = words
		self.root = Node( depth = 0, parent = None )
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth )
		# self.root.printNode()

class Node:
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.history = []
	
	def get_query( self ):
		return self.query_idx
	
	def get_child( self, response ):
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	def process_leaf( self, my_words_idx, history ):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	def process_node( self, all_words, my_words_idx, history ):
		split_dict = {}
		query_idx = 0
		if len( history ) == 0:
			query_idx = my_words_idx[np.random.randint( 0, len( my_words_idx ) )]
			query = all_words[ query_idx ]

			for word_len in range ( 4, 16 ):
				for idx in my_words_idx:
					mask = ( '_ ' * (word_len-1) )
					mask += '_'
					if len( all_words[ idx ] ) == word_len:
						if mask not in split_dict:
							split_dict[ mask ] = []
						split_dict[ mask ].append( idx )
		else:
			query_idx = my_words_idx[np.random.randint( 0, len( my_words_idx ) )]
			query = all_words[ query_idx ]
		
			for idx in my_words_idx:
				mask = self.reveal( all_words[ idx ], query )
				if mask not in split_dict:
					split_dict[ mask ] = []
				split_dict[ mask ].append( idx )
		
		return ( query_idx, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    " ):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx, self.history )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( self.all_words, self.my_words_idx, self.history )
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				history = []
				history.append( [ self.query_idx, response ] )
				# print(history)
				self.children[ response ].history = history
				
				# Recursively train this child node
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, fmt_str )
	
	def printNode( self ):
		for i in range ( self.depth ):
			print( "    ", end = '' )
		print( self.history, end = ', ' )
		print( self.all_words[self.query_idx] )
		for child in self.children.values():
			child.printNode()

# Load the words
words = np.loadtxt( 'dict_secret', dtype = str )

# The final model
res_model = my_fit( words, words.shape[0] )
