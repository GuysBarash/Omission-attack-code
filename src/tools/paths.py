import os

not_my_data = set(dir())

work_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
src_path = os.path.join(work_path, 'src')
result_path = os.path.join(work_path, 'results_@')

if True:
    my_data = set(dir()) - not_my_data
    my_data.remove('not_my_data')
    itemized = dict()
    for k in my_data:
        itemized[k] = eval(k)
