# -*- coding: utf-8 -*-
'''
@Time    : 10/11/22 
@Author  : Zhang Haoliang
'''

import subprocess
# cmd = ["blender --background --python demo_test.py -- --input=./dfaustdata/5_1237.obj --output=./testAutocall.png"]
# import os
# os.system(cmd[0])
# returned_value = subprocess.Popen(cmd, shell=True)  # returns the exit code in unix
# print('returned value:', returned_value)

import os
cmdString="blender --background --python demo_head_auto.py"
cmdvcolor="blender --background --python demo_vcolor_head.py"
file="/Users/mars_zhang/Downloads/BlenderTool/paperDemoData/headdata/pred3"
normalblender=[]
errorblender=[]
for name in os.listdir(file):
       if name.endswith('obj'):
              s=cmdString+" -- --input="+"'./headdata/pred3/"+name+"'"+" --output='./headdata/pred3/"+name[:-4]+".png'"
              normalblender.append(s)
       elif name.endswith('ply'):
              s = cmdvcolor + " -- --input=" + "'./headdata/pred3/" + name + "'" + " --output='./headdata/pred3/" + name[:-4] + ".png'"
              errorblender.append(s)

normalblender.sort()
errorblender.sort()
f = open("cmdFile3.txt", 'w')
for s in normalblender:
       f.write(s)
       f.write("\n")
for s in errorblender:
       f.write(s)
       f.write("\n")
f.close()

