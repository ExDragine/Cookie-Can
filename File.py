import os
import requests

char = 'https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_'
char2 = '.pdf'
file = open('get.cmd', 'w+')
lis1 = []

for i in range(1, 10):
    charFinal = 'wget ' + char + '0' + str(
        i) + char2 + ' -o GAMES101_Lecture_' + str(i) + '.pdf' + '\n'
    lis1.append(charFinal)
for i in range(10, 23):
    charFinal = 'wget ' + char + str(
        i) + char2 + ' -o GAMES101_Lecture_' + str(i) + '.pdf' + '\n'
    lis1.append(charFinal)
file.writelines(lis1)
