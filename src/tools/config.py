not_my_data = set(dir())

# type of workload handling (concurrent, parallel)
workload_handling = 'concurrent'

# Random seed, if "None" a random seed will be randomized
random_seed = None

# Stop when attack is successful
stop_on_success = True

# Store results or re-evaluate each time
store_results = False

if True:
    my_data = set(dir()) - not_my_data
    my_data.remove('not_my_data')
    itemized = dict()
    for k in my_data:
        itemized[k] = eval(k)
