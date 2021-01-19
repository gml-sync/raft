if [[ ! -d $TRAINHOME/git/runs ]] ; then
	echo 'Path invalid, aborting'
	exit
fi
cd $TRAINHOME
rm out/*
cd $TRAINHOME/git/runs
rm *.log
rm logbook/*
