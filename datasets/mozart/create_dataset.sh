mkdir source
cd source
wget "https://archive.org/compress/MozartCompleteWorksBrilliant170CD/formats=OGG%20VORBIS&file=/MozartCompleteWorksBrilliant170CD.zip"
unzip \*.zip
rm *.zip
for file in *.ogg; do ffmpeg -y -i "$file" -ac 1 -ar 16000 "${file%.ogg}.wav" && rm "$file"; done
cd ..
mkdir data/
th ../../scripts/generate_dataset.lua -source_path source/ -dest_path data/
rm -r source/
