from __future__ import division, print_function, unicode_literals
import io
import json
import numpy as np
import matplotlib.pyplot as plt



def plot(file_name):
    with io.open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        results = json.loads(f.read())
        line, = plt.plot([i for i in xrange(len(results))], results, 'b.-')
        plt.ylabel('log likelihood', fontsize='medium', verticalalignment='center', horizontalalignment='right',
                    rotation='vertical')
        plt.xlabel('iteration')
        
    plt.show()
    






if __name__ == "__main__":
    #~ file_name = 'results_test.json'
    file_name = 'results_state_of_the_union.json'
    #~ file_name = 's3_20topics_0.1alpha_0.1beta_100iterations_gammaln.json'
    plot(file_name)
    
