import argparse
import sys
sys.path.append('/Users/mars_zhang/Downloads/BlenderTool/') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np
cwd = os.getcwd()
# default arg
def parse_arguments():
  parser = argparse.ArgumentParser('blender the image')
  parser.add_argument('--input', type=str, default='./dfaustdata/5_1237.obj')
  parser.add_argument('--output', type=str, default='./5_1237.png')
  return parser.parse_known_args(sys.argv[sys.argv.index("--")+1:])
args, unknown = parse_arguments()
outputPath = os.path.join(cwd, args.output) # make it abs path for windows

# blender --background --python 'demo_test2.py' -- --input='./dfaustdata/5_1237.obj' --output='./test.png'
# ## initialize blender
# imgRes_x = 550
# imgRes_y = 550
# numSamples = 100
# exposure = 1.5
# bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
#
# ## read mesh (choose either readPLY or readOBJ)
# meshPath = args.input
# location = (0.573,-0.012,0.59) # (UI: click mesh > Transform > Location)
# rotation = (90, 0, 90) # (UI: click mesh > Transform > Rotation)
# scale = (1,1,1) # (UI: click mesh > Transform > Scale)
# mesh = bt.readMesh(meshPath, location, rotation, scale)
#
# ## set shading (uncomment one of them)
# bpy.ops.object.shade_smooth()
#
#
#
# # # set material
# # colorObj(RGBA, H, S, V, Bright, Contrast)
# meshColor = bt.colorObj(bt.derekBlue, 0.5, 1.0, 1.0, 0.0, 2.0)
# AOStrength = 0.0
# bt.setMat_balloon(mesh, meshColor, AOStrength)
#
# ## set invisible plane (shadow catcher)
# bt.invisibleGround(shadowBrightness=0.9)
#
# ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
# camLocation = (3, 0, 2)
# lookAtLocation = (0,0,0.5)
# focalLength = 45 # (UI: click camera > Object Data > Focal Length)
# cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
#
# ## set light
# lightAngle = (6, -30, -155)
# strength = 2
# shadowSoftness = 0.3
# sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
#
# ## set ambient light
# bt.setLight_ambient(color=(0.1,0.1,0.1,1))
#
# ## set gray shadow to completely white with a threshold
# bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')
#
# ## save blender file so that you can adjust parameters in the UI
# bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
#
# ## save rendering
# bt.renderImage(outputPath, cam)