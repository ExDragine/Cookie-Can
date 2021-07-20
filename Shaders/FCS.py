# To create a shaders as fast as we can.

import os

i18n = ['en_US', 'ar_AE', 'es_ES', 'fr_FR', 'ru_RU', 'zh_CN']
OptiFine_File = ['block.properties', 'shaders.properties']
MutiList = ['shadowcomp', 'prepare', 'deferred', 'composite']
SingleList = [
    'shadow', 'gbuffers_basic', 'gbuffers_textured', 'gbuffers_textured_lit',
    'gbuffers_skybasic', 'gbuffers_skytextured', 'gbuffers_clouds',
    'gbuffers_terrain', 'gbuffers_damagedblock', 'gbuffers_block',
    'gbuffers_beaconbeam', 'gbuffers_item', 'gbuffers_entities',
    'gbuffers_entities_glowing', 'gbuffers_armor_glint', 'gbuffers_spidereyes',
    'gbuffers_hand', 'gbuffers_weather', 'gbuffers_water',
    'gbuffers_hand_water', 'final'
]
ExtName = ['.vsh', '.fsh', '.gsh', '.csh']
Folder = [
    'Function', 'Lib', 'Lang', 'Model', 'Resource', 'world0', 'world-1',
    'world1'
]
ExtFolder = ['Doc']
Syntax = ['Vertex', 'Fragment', 'Geometry', 'Compute', 'Logo']
Raw_Data = []

# os.chdir(r'C:\Users\surfa\Desktop\Minecraft\VisionLab\.minecraft\shaderpacks\Horizon\shaders')
PRN = os.getcwd()

Path = os.getcwd() + r'\Starry\shaders'

for Data in Syntax:
    Raw = open(Data, 'r+')
    Raw_Data.append(Raw.readlines())
    Raw.close()

Version = ['#version 460 Core', '\n' * 2]
Logo = Raw_Data[4]
Program_Data = [Raw_Data[0], Raw_Data[1], Raw_Data[2], Raw_Data[3]]

if os.path.exists(Path):
    print('光影目录已存在')
else:
    os.makedirs(Path)
    print('光影目录创建成功')

os.chdir(os.getcwd() + r'\Starry\shaders')

for i in Folder:
    if os.path.exists(i):
        print('子目录', i, '已存在')
        continue
    else:
        os.makedirs(i)
        print(i, '创建成功')

for Var in OptiFine_File:
    Works = open(Var, 'w+')
    Works.write('#OptiFine配置文件，可根据doc内文件修改')
    Works.close()

for i in range(len(ExtName)):
    for Var in SingleList:
        Works = open(Var + ExtName[i], 'w+')
        Works.writelines(Version + Logo + Program_Data[i])
        Works.close()
for i in range(len(ExtName)):
    for Var in MutiList:
        Works = open(Var + ExtName[i], 'w+')
        Works.writelines(Version + Logo + Program_Data[i])
        Works.close()
        for o in range(1, 16):
            File_Name = Var + str(o)
            Works = open(File_Name + ExtName[i], 'w+')
            Works.writelines(Version + Logo + Program_Data[i])
            Works.close()

os.chdir(PRN + r'\Starry\shaders\Lang')

for i in i18n:
    Works = open(i + '.lang', 'w+')
    Works.write('#本文件根据联合国工作用语言确定')
    Works.close()

os.chdir(PRN + r'\Starry\shaders\Lib')
Works = open('Universal.glsl', 'w+')
Works.close()
