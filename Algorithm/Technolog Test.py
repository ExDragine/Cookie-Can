import requests
import json

MinecraftVersion = json.loads(requests.api.get('http://launchermeta.mojang.com/mc/game/version_manifest.json').text)
MinecraftVersion = str(MinecraftVersion['latest']['release'])
MinecraftVersion = MinecraftVersion[:4]
print(MinecraftVersion)
PaperURL = 'https://papermc.io/api/v2'

Projects = json.loads(requests.api.get(PaperURL+'/projects').text)
Projects = Projects['projects']

VersionGroup = json.loads(requests.api.get(PaperURL+'/projects/paper/version_group/'+MinecraftVersion+'/builds').text)
version = VersionGroup['versions'][-1]
build = VersionGroup['builds'][-1]['build']
download = VersionGroup['builds'][-1]['downloads']['application']['name']

tmp = requests.api.get(PaperURL+'/projects'+'/paper/versions/'+str(version)+'/builds/'+str(build)+"/downloads/"+str(download))
file = open('paper.jar','wb+')
file.write(tmp.content)
file.close()
