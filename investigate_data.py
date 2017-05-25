import matplotlib.pyplot as plt
from util import *

def custom_transform(X):
    (num_elems, dimensions) = np.shape(X)
    new_X = []
    for n in xrange(num_elems):
        x = X[n]
        nX = np.array([x[0], x[1], x[2], x[3], x[4], x[5], pow(x[6], 0.5), pow(x[7], 0.5), pow(x[8], 0.5), pow(x[9], 0.5)])
        new_X.append(nX)
    new_X_np = np.asarray(new_X)
    return new_X_np

def remove_outliers(X):
    (num_elems, dimensions) = np.shape(X)
    new_X = []
    for n in xrange(num_elems):
        x = X[n]
        for d in xrange(dimensions):
            if x[d] > 2:
                x[d] = 1
        new_X.append(x)
    new_X_np = np.asarray(new_X)
    return new_X_np

def separate_intervals(Y):
    dimensions = np.shape(Y)[0]
    new_Y = []
    for d in xrange(dimensions):
        nY = int(Y[d])/5
        new_Y.append(nY)
    return np.array(new_Y).ravel()

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
X_t = custom_transform(n_X)
[mins, maxes] = get_max_and_mins(X_t)

n_X_scaled = scale_vars(X_t, mins, maxes)
n_X_std = remove_outliers(n_X_scaled)
n_Y = separate_intervals(Y)

correlations = test_correlations(n_X_scaled, Y)

(num_elems, dimensions) = np.shape(n_X)
print 'Saving scatter plots'

for d in xrange(dimensions):
    print 'feature', d + 1, 'of', dimensions
    feature = isolate_feature(n_X_std, d)
    plt.gcf().clear()
    plt.xlabel('feature ' + str(d))
    plt.ylabel('saida')
    plt.scatter(feature, n_Y)
    plt.savefig('result_images/scatter_' + str(d) + '1_.png')

#plt.gcf().clear()
#plt.scatter((isolate_feature(n_X_scaled, 3))/(0.0001 + isolate_feature(n_X_scaled, 9)), Y, color='blue')
#plt.scatter(isolate_feature(n_X_scaled, 3), Y, color='red')
#plt.show()
