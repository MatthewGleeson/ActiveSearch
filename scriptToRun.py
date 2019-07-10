#from active_search import ActiveLearning
from active_search.active_search import ActiveLearning






learner = ActiveLearning(visual=True,random=False)
learner.run(1000)
