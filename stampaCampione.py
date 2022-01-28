from pydriller import Repository
import xlsxwriter
from urlRepo import repo
from random import *
import os


workbook = xlsxwriter.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet()

i = 1
listaCommit = []

for r in repo:
    try:
        for commit in Repository(r).traverse_commits():
            listaCommit.append(commit.msg+" , "+r)
    except:
        listaCommit.append("####")

for i in range(2001):
    rand = randint(1, 684589) 
    worksheet.write(f'A{i}', listaCommit[rand])
    print(listaCommit[rand])

workbook.close()
os.system("pause")