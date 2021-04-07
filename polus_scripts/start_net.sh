#tar  --skip-old-files -xf FlyingThings3D_subset_image_clean.tar.bz2
cd $TRAINHOME/git
#python -u train.py --name raft-chairs --stage chairs --validation chairs --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 --num_steps 250000 --batch_size 4 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --output $OUTPUTS >>$OUTPUTS/main_stdout.log 2>>$OUTPUTS/main_stderr.log
#python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --output $OUTPUTS >>$OUTPUTS/main_stdout.log 2>>$OUTPUTS/main_stderr.log
#python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 250000 --batch_size 4 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --output $OUTPUTS >>$OUTPUTS/main_stdout.log 2>>$OUTPUTS/main_stderr.log
python evaluate.py --model=checkpoints/raft-things.pth --output=$OUTPUTS --dataset=sintel >>$OUTPUTS/main_stdout.log 2>>$OUTPUTS/main_stderr.log
