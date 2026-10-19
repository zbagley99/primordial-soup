#!/usr/bin/env python3
"""Independently probe and decode our small video acceptance fixtures."""
import json, subprocess, sys
from pathlib import Path
reports=[]
for filename in sys.argv[1:]:
    path=Path(filename)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)]))
    stream=probe['streams'][0]
    hashes=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-threads','1','-f','framemd5','-'],text=True)
    decoded=[line.split(',')[-1].strip() for line in hashes.splitlines() if line and not line.startswith('#')]
    assert len(decoded)==int(stream['nb_read_frames'])
    assert len(set(decoded))>1, 'Video contains only identical frames'
    middle=(len(decoded)-1)//2
    proof=path.with_name(path.stem+'-frames.png')
    select=f'select=eq(n\\,0)+eq(n\\,{middle})+eq(n\\,{len(decoded)-1}),tile=3x1'
    subprocess.run(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-i',str(path),'-vf',select,'-frames:v','1','-threads','1','-y',str(proof)],check=True)
    reports.append({'file':str(path),'bytes':path.stat().st_size,'probe':probe,'decoded_frames':len(decoded),'unique_decoded_frames':len(set(decoded)),'proof':str(proof)})
print(json.dumps(reports,indent=2))
