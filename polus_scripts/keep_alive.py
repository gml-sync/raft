import time
from datetime import datetime
import os
import subprocess
import argparse

# Usage: keep_alive.py <--expid 1> [--single]

def subp_run_str(cmd, output=True):
    print('RUN:', cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        
    if output:
        for line in process.stdout:
            print(line.decode(), end='')
    rc = process.poll()
    return rc

def subp_bash(cmd):
    print('RUN:', cmd)
    os.system(f'bash -c "{cmd}"')
    #subp_run_str("bash -c '" + cmd + "'")

def datestr():
    now = datetime.now()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

# even if script is called from different folder, switch to containing folder
parser = argparse.ArgumentParser()
parser.add_argument('--expid', help="experiment name")
parser.add_argument('--single', action='store_true', help="launch job only once")
args = parser.parse_args()

print(f"experiment name={args.expid}\nsingle run={args.single}")
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#output = os.environ['OUTPUTS'] # this will raise an error because it's a dictionary
file_path = f'{args.expid}/job_stderr.txt'
if os.path.exists(file_path):
    os.remove(file_path)

while True:
    print(datestr(), args.expid, 'submit')
    subp_bash(
        f'bsub -o {args.expid}/job_stdout.txt '
        f'-e {args.expid}/job_stderr.txt -W 02:30 -q normal -gpu "num=1:mode=exclusive_process" '
        f'bash $TRAINHOME/git/polus_scripts/start_net.sh {args.expid}')
    if args.single:
        print('single run')
        break
    print('wait...')
    while not os.path.exists(file_path):
        time.sleep(5)
    os.remove(file_path)
    