from active_search.active_search import ActiveLearning
import time

start_time = time.time()
learner = ActiveLearning(visual=True,jitter=True,random=False,utility = 3, do_pruning = True)
learner.run(100)
print("--- %s seconds ---" % (time.time() - start_time))
