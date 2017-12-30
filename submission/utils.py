# a decorator that prints the function's name
def print_name(func):
	def func_printed(*args):
		print "Running %s." % func.__name__
		func(*args)
	return(func_printed)