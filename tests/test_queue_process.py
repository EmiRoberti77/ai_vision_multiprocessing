from multiprocessing import Process, Queue
import time

def producer(q:Queue)->None:
    for i in range(5):
        print(f"Producing {i}")
        q.put(i)


def consumer(q:Queue)->None:
    while True:
        item = q.get()

        if item is None:
            break
    
        for i in range(5):
            time.sleep(1)
            print(f"processing-iten={item}:i={i}")

        print(f"completed item {item}")


queue = Queue()

p1 = Process(target=producer, args=(queue,))
p2 = Process(target=consumer, args=(queue,))

p1.start() #start the producer process
p2.start() #start the consumer process


p1.join() #wait for that to finish to start the next process
p2.join() #wait for them to finish
