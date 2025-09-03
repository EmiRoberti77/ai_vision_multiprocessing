from ast import arg
from multiprocessing import Process, Manager
from time import sleep

def update_shared_state(shared_dict:Manager):
    for i in range(5):
        key = f"key_{i}"
        value = i * 10
        shared_dict[key] = value
        print(f"updated {key} to {value}")
        sleep(1)


if __name__== "__main__":
    with Manager() as manger:
        shared_dict = manger.dict()
        p = Process(target=update_shared_state, args=(shared_dict,))
        p.start()
        print("function()=1")
        p.join()
        print("function()=2")