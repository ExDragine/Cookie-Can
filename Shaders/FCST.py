import os

i18n = ['en_US', 'ar_AE', 'es_ES', 'fr_FR', 'ru_RU', 'zh_CN']
OptiFine_File = ['block.properties', 'shaders.properties']
Shaders = ['shadowcomp', 'prepare', 'deferred', 'composite']
ShaderName = [
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

Name = input('请输入你的光影名字\n')
OGLV = input('请输入GLSL语言规范版本\n')
OGLM = input('请输入GLSL语言模式\n')

Version = ['#version ' + OGLV + ' ' + OGLM, '\n'*2]

RNP = os.getcwd()  # RNP: Right Now Path


def FileWorker(File, Data, Model):
    Worker = open(File, Model)
    Worker.writelines(Data)
    Worker.close()


def DataLoad(input):
    DataStream = []
    for Data in input:
        Worker = open(Data, 'r+')
        DataStream.append(Worker.readlines())
        Worker.close()
    return DataStream


def DirWorker(Path):
    for Worker in Path:
        if os.path.exists(Path):
            print('目录'+Path+'已存在')
            continue
        else:
            os.makedirs(Path)


def main(Name):
    DataStream = DataLoad(Syntax)
    ProgramData = DataStream.pop()

    DirWorker(RNP + '/' + Name + '/' + 'shaders')
    os.chdir(RNP + '/' + Name + '/' + 'shaders')
    for Worker in Folder:
        DirWorker(Worker)

    for Worker in OptiFine_File:
        FileWorker(Worker, list('#OptiFine配置文件，可根据doc内文件修改'), 'w+')

    for i in range(len(ExtName)):
        for Name in ShaderName:
            FileWorker(File=Name+ExtName[i], Data=Version +
                       DataStream[4]+ProgramData[i], Model='w+')
        for Name in Shaders:
            FileWorker(File=Name+ExtName[i], Data=Version +
                       DataStream[4]+ProgramData[i], Model='w+')
            for o in range(1, 16):
                FileWorker(
                    File=Name+str(o)+ExtName[i], Data=Version+DataStream[4]+ProgramData[i], Model='w+')
    os.chdir(RNP + '/' + Name + '/' + 'shaders/lang')
    for Worker in i18n:
        FileWorker(File=Worker+'.lang',
                   Data=list('#本文件根据联合国工作用语言确定'), Model='w+')
    os.chdir(RNP + '/' + Name + '/' + 'shaders/Lib')
    FileWorker(File='Universal.glsl', Data=list('#预留文件'), Model='w+')


main(Name)
