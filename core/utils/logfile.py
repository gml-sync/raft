from datetime import datetime

class Logfile:
	logfile_name = './log.txt'
	logfile = None

	def __init__(self, filename):
		self.logfile_name = filename
		self.logfile = open(self.logfile_name, 'a')
		print('Log object created')

	def log(self, *args, **kwargs):
		print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), end='   ', file=self.logfile)
		print(*args, file=self.logfile, **kwargs) # Pass on all positional and keyword arguments

	def __del__(self):
		self.logfile.close()

logfile = Logfile('log.txt')