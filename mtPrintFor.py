import sys
import threading
import time
from pydriller import Repository
from listaRepo import repo

commit_list = list()


def thread_function(repository, index):
    count = 0
    try:
        for commit in Repository(repository).traverse_commits():
            count += 1
        commit_list.append(f'{index};{repository};{count}')
    except:
        commit_list.append(f'{index};{repository};Error')
    print(f'counting: {repository}')


def repo_with_thread(start):
    threads = list()
    for index, repo_link in enumerate(repo):
        x = threading.Thread(target=thread_function, args=(repo_link, index,))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()
    print(time.time() - start)


if __name__ == "__main__":
    start_time = time.time()
    repo_with_thread(start_time)

    original_stdout = sys.stdout
    with open('numCommit.csv', 'w') as f:
        sys.stdout = f
        for commit in commit_list:
            print(commit)
        sys.stdout = original_stdout

    # repo_without_thread(start_time)