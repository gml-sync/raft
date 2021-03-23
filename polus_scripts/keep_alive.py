import time
from datetime import datetime
import os
import subprocess
import argparse

def subp_run_str(cmd, output=True):
    print('RUN:', cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        
    if output:
        for line in process.stdout:
            print(line.decode(), end='')
    rc = process.poll()
    return rc

def subp_bash(cmd):
    subp_run_str("bash -c '" + cmd + "'")

def datestr():
    now = datetime.now()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

os.chdir(os.path.dirname(os.path.realpath(__file__)))
output = os.environ['OUTPUTS'] # this will raise an error because it's a dictionary
file_path = '{}/job_stderr.txt'.format(output)
if os.path.exists(file_path):
    os.remove(file_path)

while True:
    print(datestr() + ' submit')
    subp_bash('bash ./submit.sh')
    print('wait...')
    while not os.path.exists(file_path):
        time.sleep(5)
    os.remove(file_path)
    