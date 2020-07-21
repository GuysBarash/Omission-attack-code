import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

not_my_data = set(dir())

# Input generation:
samples = 200
dimensions = 2
clusters = 2
test_size = 0.5

type_of_inputs = dict()
type_of_inputs['2_blobs'] = {
    'tag': '2_blobs',
    'cluster_std': 3,
    'centers': [(-3, -3), (3, 3), (-3, 3), (3, -3)],
}
type_of_inputs['poly'] = {
    # The line represented is (P2) = a + b(P1) + c(P2^2) + d(P1^3)
    'tag': 'poly',
    'range': (-10, 10),
    'a': 3,
    'b': -1,
    'c': 0,
    'd': 0,

}

# Chosen data format
type_of_input = type_of_inputs['2_blobs']

# Classefier
clfs = dict()
clfs['SVM_linear'] = ('SVM_linear', SVC(kernel="linear", probability=True))
clfs['SVM_gamma'] = ('SVM_linear', SVC(gamma=2, C=1, probability=True))
clfs['GaussianProcess'] = ('GaussianProcess', GaussianProcessClassifier(1.0 * RBF(1.0)))
clfs['DTree'] = ('DTree', DecisionTreeClassifier())
clfs['RandomForest'] = ('RandomForest', RandomForestClassifier())
clfs['MLP'] = ('MLP', MLPClassifier(alpha=1))
clfs['KNN5'] = ('KNN5', KNeighborsClassifier(n_neighbors=5))
clfs['KNN3'] = ('KNN3', KNeighborsClassifier(n_neighbors=3))
clfs['KNN1'] = ('KNN1', KNeighborsClassifier(n_neighbors=1))
clfs['AdaBoost'] = ('AdaBoost', AdaBoostClassifier())
clfs['GaussianNB'] = ('GaussianNB', GaussianNB())
clfs['ANN'] = ('ANN', MLPClassifier(hidden_layer_sizes=(4, 4,)))

# Chosen clf
clf = clfs['SVM_linear']

# Adversarial sample placement
adversarial_tactics = dict()
adversarial_tactics['std_ratio'] = ['std_ratio',
                                    {
                                        'DISTANCES': (7, 10),  # distance from red, distance from blue
                                        'HEIGHT_RATIO': None,
                                        'MAX_HEIGHT': 25,
                                    }]

adversarial_tactic = adversarial_tactics['std_ratio']

# Attack
min_budget = 1
max_budget = int(np.ceil(np.sqrt(samples * clusters)))
budget_step = 1
budget_range = [max_budget]  # range(min_budget, max_budget, budget_step)

aggregation_functions = dict()
aggregation_functions['mean'] = {'key': 'mean',
                                 'aggreg_func_blue': np.mean,
                                 'aggreg_func_red': np.mean,
                                 }
aggregation_functions['optimistic'] = {'key': 'optimistic',
                                       'aggreg_func_blue': np.max,
                                       'aggreg_func_red': np.min,
                                       }
aggregation_functions['pessimistic'] = {'key': 'pessimistic',
                                        'aggreg_func_blue': np.min,
                                        'aggreg_func_red': np.max,
                                        }
aggregation_function = aggregation_functions['optimistic']

repetitions = 1

attack_tactics = dict()
attack_tactics['by_distance_from_adv'] = ('by_distance_from_adv',
                                          {
                                              'budget_for_remove_red': 1,
                                              'budget_for_remove_blue': 0,
                                          })

attack_tactics['greedy_search'] = ('greedy_search',
                                   {
                                       'iteration_budget': 3,
                                   })

attack_tactics['Genetic'] = ('Genetic',
                             {
                                 'generations': 1200,
                                 'offsprings': 40,
                                 'parents': 3,
                                 'KNN_parent': True,
                             })

attack_tactic = attack_tactics['greedy_search']
plot_contour = False

if True:
    my_data = set(dir()) - not_my_data
    my_data.remove('not_my_data')
    itemized = dict()
    for k in my_data:
        itemized[k] = eval(k)
