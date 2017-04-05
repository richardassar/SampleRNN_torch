mkdir source/
wget -r -H -nc -nH --cut-dir=1 -A .ogg -R *_vbr.mp3 -e robots=off -P source/ -l1 -i ./itemlist.txt -B 'http://archive.org/download/'
mv source/*/*.ogg source/
rm -r source/*/
for file in source/*.ogg; do ffmpeg -i "$file" -ac 1 -ar 16000 "source/`basename $file .ogg`.wav" && rm $file; done
mkdir data/
th ../../scripts/generate_dataset.lua -source_path source/ -dest_path data/
rm -r source/