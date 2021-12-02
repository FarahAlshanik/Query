import multiprocessing

def mp_worker(number):
    return number



'''
def mp_handler():
    p = multiprocessing.Pool(32)
    numbers = list(range(1000))
    with open('results.txt', 'w') as f:
        for result in p.imap(mp_worker, numbers):
            f.write('%d\n' % result)

if __name__=='__main__':
    mp_handler()





'''

def similarities_vectorized(vector_data):

    return vector_data


def mp_handler():
    p = multiprocessing.Pool(32)
    i=0
    numbers = list(range(1000))
    with open('results.txt', 'w') as f:
        for result in p.apply(mp_worker,args=(i)):
            f.write('%d\n' % result)
            i+=1


'''
def mp_handler():
#    p = multiprocessing.Pool(32)
    numbers = list(range(1000))
    for i in range(int(len(numbers)/2)):
    #print(i)
        p1 = multiprocessing.Process(target=similarities_vectorized,args=((str(i))))
        p1.start()
        with open('results.txt', 'w') as f:
            for result in p1:
                f.write('%s\n' % result)

if __name__=='__main__':
    mp_handler()


'''
'''

def create_averaging_process(processes ,q):
    p = multiprocessing.Process(target=get_averages, args=(q))
    processes.append(p)
    p.start()
 
def write_to_file(file, q, end):
    with open(file, 'a') as f:
        while True: 
            line = q.get()
            if line == end:
                return
            f.write(str(line))
             
 
def get_averages(q):

    x=5
    try:
        q.put(x)
    except Exception as e:
        print(e)
 
 
if __name__ == "__main__":
    index = 0
    processes = []
    queue = multiprocessing.Queue()
 
    STOP_TOKEN="end"
     
    i=0
    while (i<5):
        create_averaging_process(processes, queue)
        i+=1
    
    writer_process = multiprocessing.Process(target = write_to_file, args=("test.txt", queue, STOP_TOKEN))
    writer_process.start()
     
    queue.put(STOP_TOKEN)
    writer_process.join()
'''

s=[]
import multiprocessing 
  
def square_list(mylist, q): 
    """ 
    function to square a given list 
    """
    # append squares of mylist to queue 
    for num in mylist: 
        q.put([num * num,num]) 
  
def print_queue(q): 
    """ 
    function to print queue elements 
    """
    global s
    print("Queue elements:")
    dis=open("tt.txt",'w') 
    while not q.empty(): 
        #print(q.get())
        #dis.write('%s\n' %str(q.get()))
        for i in q.get():
            dis.write('%s ' % i)
        dis.write('\n')
    dis.close() 
    print("Queue is now empty!") 
  
if __name__ == "__main__": 
    # input list 
    mylist = [1,2,3,4] 
  
    # creating multiprocessing Queue 
    q = multiprocessing.Queue() 
    # creating new processes 
    p1 = multiprocessing.Process(target=square_list, args=(mylist, q)) 
    p2 = multiprocessing.Process(target=print_queue, args=(q,)) 
    
    # running process p1 to square list 
    p1.start() 
    p1.join() 
  
    # running process p2 to get queue elements 
    p2.start() 
    p2.join() 
    print(s)
