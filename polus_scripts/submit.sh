bsub -o $TRAINHOME/out/stdout_val.txt \
    -e $TRAINHOME/out/stderr_val.txt -W 02:30 -q normal -gpu "num=2:mode=exclusive_process" \
    bash $TRAINHOME/start_net.sh