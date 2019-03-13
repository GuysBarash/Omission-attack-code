not_my_data = set(dir())

# type of workload handling (concurrent, parallel)
workload_handling = 'parallel'

# Random seed, if "None" a random seed will be randomized
random_seed = None

if True:
    my_data = set(dir()) - not_my_data
    my_data.remove('not_my_data')
    itemized = dict()
    for k in my_data:
        itemized[k] = eval(k)
