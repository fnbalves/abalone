import matplotlib.pyplot as plt
from util import *

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
