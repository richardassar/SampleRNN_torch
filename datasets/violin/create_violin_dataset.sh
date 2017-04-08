mkdir source
cd source
wget "https://archive.org/compress/Op.124CapricesForSoloViolin/formats=OGG%20VORBIS&file=/Op.124CapricesForSoloViolin.zip"
wget "https://archive.org/compress/213PartitaNo.2Chaconne/formats=OGG%20VORBIS&file=/213PartitaNo.2Chaconne.zip"
wget "https://archive.org/compress/110SonateNo.3EnUtMajeurPour/formats=OGG%20VORBIS&file=/110SonateNo.3EnUtMajeurPour.zip"
unzip \*.zip
rm *.zip
for file in *.ogg; do ffmpeg -y -i "$file" -ac 1 -ar 16000 "${file%.ogg}.wav" && rm "$file"; done
cd ..
mkdir data/
th ../../scripts/generate_dataset.lua -source_path source/ -dest_path data/
rm -r source/
