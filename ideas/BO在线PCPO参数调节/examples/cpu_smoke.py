from bo_tuner import PCPOParameterBO

tuner = PCPOParameterBO(warmup_trials=2, kl_budget=.05)
for i in range(5):
    choice = tuner.suggest([i / 5, .01 * i, .2])
    tuner.observe(.02 * i - .001 * choice["learning_rate"] * 1000, .01 + .005 * i)
print("parameter-tuning smoke passed", len(tuner.history), tuner.suggest([1, 0, 0]))
