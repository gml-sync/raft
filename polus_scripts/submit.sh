bsub -o $OUTPUTS/job_stdout.txt \
    -e $OUTPUTS/job_stderr.txt -W 02:30 -q normal -gpu "num=1:mode=exclusive_process" \
    bash $TRAINHOME/git/polus_scripts/start_net.sh