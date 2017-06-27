import pickle
import numpy as np

knn_accuracies = pickle.load(open('result_pickles/scores_knn.pickle', 'rb'))
svm_accuracies = pickle.load(open('result_pickles/scores_svm.pickle', 'rb'))
mlp_accuracies = pickle.load(open('result_pickles/scores_mlp.pickle', 'rb'))
tree_accuracies = pickle.load(open('result_pickles/scores_tree.pickle', 'rb'))

all_accuracies = []

for i, k in enumerate(knn_accuracies):
    print knn_accuracies[i]
    new_data = {'knn_accuracies': knn_accuracies[i], 'svm_accuracies': svm_accuracies[i], 'mlp_accuracies': mlp_accuracies[i], 'tree_accuracies': tree_accuracies[i]}
    all_accuracies.append(new_data)

accuracies_together = []

tam_test = 30*3
num_classifiers = 4
ranks_matrix = []


def find_index(array_var, title):
    for i, e in enumerate(array_var):
        if e[0] == title:
            return i + 1
    return -1

for a in all_accuracies:
    size_k_fold = len(a['knn_accuracies'])
    
    for i in xrange(size_k_fold):
        knn = a['knn_accuracies'][i]
        mlp = a['mlp_accuracies'][i]
        svm = a['svm_accuracies'][i]
        tree = a['tree_accuracies'][i]

        results_coupled = [['knn', knn], ['mlp', mlp],
                           ['svm', svm], ['tree', tree]]
        
        rs = sorted(results_coupled, key= lambda x: x[1], reverse=True)
        new_line = [find_index(rs, 'knn'), find_index(rs, 'mlp'),
                    find_index(rs, 'svm'), find_index(rs, 'tree')]
        ranks_matrix.append(new_line)

ranks_np = np.array(ranks_matrix)
mean_ranks = np.mean(ranks_np, axis=0)

k = float(num_classifiers) - 1

sum_squared_ranks = float(np.sum(pow(mean_ranks, 2)))

chi_square = ((12.0*tam_test)/(k*(k+1)))*(sum_squared_ranks - k*pow(k+1,2)/4.0)
statistic = chi_square
print 'tam_test', tam_test
print 'k', k
print 'chi_square', chi_square
print 'Statistic', statistic

compare_val = 7.815

if statistic > compare_val:
    print 'The classifiers are different'
else:
    print 'The classifiers are equal'

#Nemenyi test
q0 = 2.569
critical_val = q0*np.sqrt(k*(k+1)/(6*float(tam_test)))
print 'Critical values for Nemenyi test', critical_val

name_of_classifiers = ['knn', 'mlp', 'svm', 'tree']
for i, n1 in enumerate(name_of_classifiers):
    for j , n2 in enumerate(name_of_classifiers):
        if i != j:
            print 'Is', n1, 'different from', n2, '?', abs(mean_ranks[i] - mean_ranks[j]) > critical_val, 'diff', abs(mean_ranks[i] - mean_ranks[j])
