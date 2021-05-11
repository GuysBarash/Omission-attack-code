bb = '''
Dataset	clss	Surrogate	victim	method	Results
CIFAR10	10	Googlenet	Resnet18	KNN	0.25
CIFAR10	10	Googlenet	MobleNetV2	KNN	0.15
CIFAR10	10	Googlenet	VGG11	KNN	0.14
CIFAR10	10	Googlenet	AlexNet	KNN	0.08
MNIST	2	X	KNN5	KNN	0.85
MNIST	2	X	GNB	KNN	0.80
MNIST	3	X	ANN	KNN	0.69
MNIST	2	ANN	ANN	Greedy	0.46
MNIST	2	X	ANN	KNN	0.45
MNIST	3	GNB	Dtree	Genetic	0.44
MNIST	2	GNB	GNB	Genetic	0.43
MNIST	2	GNB	GNB	Greedy	0.43
MNIST	2	ANN	ANN	Genetic	0.27
MNIST	2	X	SVM	KNN	0.18
MNIST	3	X	Dtree	KNN	0.16
MNIST	2	SVM	SVM	Genetic	0.15
MNIST	3	X	KNN5	KNN	0.15
MNIST	3	X	SVM	KNN	0.10
MNIST	2	SVM	SVM	Greedy	0.07
Synthetic	2	X	KNN5	KNN	1.00
Synthetic	2	X	Dtree	KNN	0.90
Synthetic	2	X	ANN	KNN	0.65
Synthetic	2	X	SVM	KNN	0.48
Synthetic	2	X	GNB	KNN	0.17
'''

wb = '''
Dataset	classes	victim	method	Results
IMDB	2	1DConvNet	Genetic	0.80
MNIST	2	ANN	Genetic	1.00
MNIST	2	GNB	Genetic	1.00
MNIST	2	GNB	Greedy	1.00
MNIST	3	ANN	Genetic	1.00
MNIST	3	SVM	Genetic	1.00
MNIST	2	KNN5	Genetic	0.90
MNIST	2	SVM	Genetic	0.82
MNIST	3	KNN5	Genetic	0.55
MNIST	2	ANN	Greedy	0.54
MNIST	2	KNN5	Greedy	0.25
MNIST	2	SVM	Greedy	0.05
Synthetic	2	KNN5	Genetic	0.99
Synthetic	2	ANN	Genetic	0.88
Synthetic	2	SVM	Genetic	0.87
Synthetic	2	ANN	Greedy	0.86
Synthetic	2	Dtree	Genetic	0.85
Synthetic	2	GNB	Genetic	0.58
Synthetic	2	Dtree	Greedy	0.55
Synthetic	2	GNB	Greedy	0.52
Synthetic	2	KNN5	Greedy	0.36
Synthetic	2	SVM	Greedy	0.17

'''


def format_table(s):
    ls = s.split('\n')
    ls = [lt.split('\t') for lt in ls if lt != '']

    ltitle = ls[0]
    ls = ls[1:]
    print(r'\hline')
    print(' & '.join(ltitle) + '\t' + r'\\')
    print(r'\hline')
    for lt in ls:
        print(' & '.join(lt) + '\t' + r'\\')
    print(r'\hline')


format_table(wb)