#!/usr/bin/python
# Simple tokenizer that can work with different token separators
# and provides some quality of life functions such as seek and peek
class Tokenizer:
	# The tokens can be either in a string or they can be provided as a list
	def __init__(self, tokens, sep=' '):
		if isinstance(tokens, str):
			self.tokens = " ".join(tokens.split(sep)).split()
		elif isinstance(tokens, list):
			self.tokens = tokens	
		else:
			raise(TypeError("tokens must be either a string or a list"))		
		self.index = 0
		
	# returns the next token and increments the token index	
	def next_token(self):
		if(self.index >= len(self.tokens)):
			return ""
			
		ntoken = self.tokens[self.index]
		self.index+=1		
		return ntoken
	
	# return the token at position index	
	def token_at(self, index):
		return self.tokens[index]
	
	# returns false if the token index is at the end of the tokens list
	def has_more_tokens(self):
		return self.index < len(self.tokens)
	
	# move the token index to the position index
	def move_to(self, index):
		self.index = index
	
	# looks for a token equal to the token argument.
	# if a match is found the index is moved to the match index otherwise
	# the function returns -1	
	def seek(self, token):
		moved = -1
		for idx,val in enumerate(self.tokens):
			if(token == val):
				self.move_to(idx)
				moved = idx
		
		return moved
	# returns the current token without incrementing the index
	def peek(self):
		if(self.index  >= len(self.tokens)):
			return ""		
		else:
			return self.tokens[self.index]
	
	# returns an integer or raises an execption if it does not find one
	def getInt(self):
		return int(self.next_token())
		
	def getFloat(self):
		return float(self.next_token())
		
	def __str__(self):
		return "Tokenizer : at index {0} of {1} ".format(self.index,self.tokens)
	
	def getTokens(self):
		return self.tokens
