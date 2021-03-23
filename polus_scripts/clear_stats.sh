if [[ ! -d $OUTPUTS ]] ; then
	echo 'Path invalid, aborting'
	exit
fi
#cd $TRAINHOME && rm out/*
cd $OUTPUTS && rm -r ./*
