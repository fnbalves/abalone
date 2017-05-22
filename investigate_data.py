import matplotlib.pyplot as plt
from util import *

def test_correlations(X, Y):
    (num_elems, dimensions) = np.shape(X)
    correlations = []
    for i in xrange(dimensions):
        Xd = X[np.ix_(range(num_elems), [i])].ravel()
        new_correlation = np.correlate(Xd, Y)
        correlations.append(new_correlation)
        print 'correlation', i, new_correlation
    return correlations

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'
all_outputs = sorted(set(Y.tolist()))

separated_Y = []
for o in all_outputs:
    new_Y = [y for y in Y if y == o]
    separated_Y.append(new_Y)


for (i, o) in enumerate(all_outputs):
    print 'Num of class', o,':' , len(separated_Y[i])
    
plt.xlabel('idade')
plt.ylabel('frequencia amostral')
plt.hist(Y)
plt.savefig('result_images/hist_ages.png')

n_X = create_dummy_vars(X, 0)
[mins, maxes] = get_max_and_mins(n_X)
n_X_scaled = scale_vars(n_X, mins, maxes)

correlations = test_correlations(n_X_scaled, Y)