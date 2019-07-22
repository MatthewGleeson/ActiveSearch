#from active_search import ActiveLearning
from active_search.active_search import ActiveLearning




import time
start_time = time.time()

learner = ActiveLearning(visual=True,random=False,utility = 1)
learner.run(4)

print("--- %s seconds ---" % (time.time() - start_time))