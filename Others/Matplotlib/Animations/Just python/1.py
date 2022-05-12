import time
from progress.bar import IncrementalBar

mylist = range(0,1000,1)


bar = IncrementalBar('Countdown', max = len(mylist))

for item in mylist:
    bar.next()
    time.sleep(0.01)

bar.finish()
