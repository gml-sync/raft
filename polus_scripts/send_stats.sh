cd runs
zip -r logbook.zip ./*
curl -T logbook.zip ftp://staro.drevo.si:8021
rm logbook.zip
