from multiprocessing import Process
import multiprocessing


a=[]
c=[]

def func1():
  print ('func1: starting')
  for i in range(10000000000): pass
  print ('func1: finishing')
 # return_dict["1"]=12
  a.append(1)
  return 1

def func2():
  print ('func2: starting')
  for i in range(1000000000): pass
  print ('func2: finishing')
#  return_dict[0]=2
  c.append(2)
  return 2





if __name__ == '__main__':
  manager = multiprocessing.Manager()
  #return_dict = manager.dict()
  for i in range(100): 
  	p1 = Process(target=func1)
  	p1.start()
  	p2 = Process(target=func2)
  	p2.start()
  #a.append(p1)
  p1.join()
  p2.join()
  print(a)





'''

import multiprocessing

def worker(procnum,s):
    s.append(procnum)
    


if __name__ == '__main__':

    manager = multiprocessing.Manager()
    q = manager.Queue()     
    s=[]
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,s))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print (s)




from multiprocessing import Manager
from multiprocessing import Pool  
import pandas as pd

def worker(row, param):
    # do something here and then append it to row
    x = param**2
    row.append(x)

if __name__ == '__main__':
    pool_parameter = [5,6,7] # list of objects to process
    params=[]
    with Manager() as mgr:
        row = mgr.list([])

        # build list of parameters to send to starmap
        for param in pool_parameter:
            params.append([row,param])

        with Pool() as p:
            p.starmap(worker, params)

print(row)
'''

'''

import multiprocessing

def mp_worker(number):
    number += 1
    return number

def mp_handler():
    p = multiprocessing.Pool(100)
    numbers = list(range(1000))
    with open('results.txt', 'w') as f:
        for result in p.imap(mp_worker, numbers):
            f.write('%d\n' % result)

if __name__=='__main__':
    mp_handler()
'''
