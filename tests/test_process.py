from multiprocessing import Process
import time

def worker(name:str)->None:
    print(f"Worker {name} started")
    i = 0
    while i < 5:
        time.sleep(1)
        i += 1
        print(f"Worker {name} is working {i} times")
        
    print(f"Worker {name} finished")


# create two processes
if __name__ == "__main__":
    name = ['A', 'B', 'C', 'D', 'E']
    processes = []
    for i in range(5):
        p = Process(target=worker, args=(f"process {name[i]}", ))
        p.start()
        processes.append(p)


    print("Waiting for all processes to finish")
    for p in processes:
        p.join()

    print("All processes finished")