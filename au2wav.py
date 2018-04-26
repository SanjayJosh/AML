import os
from pydub import AudioSegment
allgenre=os.listdir('.')
for i  in allgenre:
    for j in os.listdir(i):
        song=AudioSegment.from_file(os.path.join(i,j),'au')
        base_name=os.path.splitext(os.path.join(i,j))[0]
        store_path=os.path.join("..","dataset",base_name+".wav")
        song.export(store_path,format='wav')
