from pydriller import Repository
from listaRepo import repo

for r in repo:
    count = 0
    try:
        for commit in Repository(r).traverse_commits():
            count += 1
        print(count)
    except:
        count = "####"
        print(count)