#from active_search import ActiveLearning
from active_search.active_search import ActiveLearning




import time
start_time = time.time()

learner = ActiveLearning(visual=True,jitter=True,random=False,utility = 2)
learner.run(100)

print("--- %s seconds ---" % (time.time() - start_time))
