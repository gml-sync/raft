from datetime import datetime

class Logfile:
	logfile_name = './log.txt'
	logfile = None

	def set_logfile(self, filename):
		self.logfile_name = filename
		self.logfile = open(self.logfile_name, 'a')

	def log(self, *args, **kwargs):
		if self.logfile:
			print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), end='   ', file=self.logfile)
			print(*args, file=self.logfile, **kwargs) # Pass on all positional and keyword arguments
			self.logfile.flush()
		else:
			exit('Log file was not set before writing')

	def __del__(self):
		if self.logfile:
			self.logfile.close()

logfile = Logfile()