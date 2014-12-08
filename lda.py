from __future__ import division, print_function, unicode_literals
import os
import io
import json
import random
import time
import math
import numpy as np
import scipy as sp
from scipy.special import gammaln


class CountMaster(object):
    def __init__(self, num_topics, alpha, beta):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.types_to_numbers = {}
        self.numbers_to_types = []
        self.type_counts = []
        self.total_tokens = 0
        # these are populated once the vocabulary stops growing
        self.topic_totals = [0 for t in xrange(num_topics)]
        self.topics = []
        
        # stores 3-tuples, format: (total topics, tokens, token numbers, token topics)
        # types (list, list, list, list)
        self.document_topic_counts = []
        self.document_token_numbers = []
        self.document_token_topics = []
        
    def add_tokens(self, tokens):
        tokens = tokens
        numbers = []
        token_topics = []
        doc_topics = [0 for t in xrange(self.num_topics)]
        for token in tokens:
            if token not in self.types_to_numbers:
                next_key = len(self.numbers_to_types)
                self.numbers_to_types.append(token)
                self.type_counts.append(0)
                self.types_to_numbers[token] = next_key
            self.type_counts[self.types_to_numbers[token]] += 1
            numbers.append(self.types_to_numbers[token])
            topic_assignment = random.randint(0, self.num_topics-1)
            token_topics.append(topic_assignment)
            doc_topics[topic_assignment] += 1
            
        self.document_topic_counts.append(doc_topics)
        self.document_token_numbers.append(numbers)
        self.document_token_topics.append(token_topics)
        self.total_tokens += len(tokens)
        #~ print(doc_topics)
        #~ print(tokens)
        #~ print(numbers)
        #~ print(token_topics)
        #~ print()
        #~ print(self.topics)
        #~ print()
    
    def done_adding_tokens(self):
        self.topics = [[0 for i in xrange(len(self.type_counts))] for t in xrange(self.num_topics)]
        for d in xrange(len(self.document_topic_counts)):
            token_numbers = self.document_token_numbers[d]
            token_topics = self.document_token_topics[d]
            for i, tok_num in enumerate(token_numbers):
                topic = token_topics[i]
                self.topic_totals[topic] += 1
                self.topics[topic][tok_num] += 1
        
        sum_tokens = 0
        sum_tokens2 = 0
        for t in xrange(self.num_topics):
            sum_tokens += self.topic_totals[t]
            for t_count in self.topics[t]:
                sum_tokens2 += t_count
        assert sum_tokens == sum_tokens2 and sum_tokens == self.total_tokens
    
    def get_num_documents(self):
        return len(self.document_topic_counts)
    
    def get_num_tokens(self, doc_index):
        return len(self.document_token_numbers[doc_index])
    
    def get_pi_vector(self, doc_index, tok_index):
        pi = []
        
        token_number = self.document_token_numbers[doc_index][tok_index]
        token_topic = self.document_token_topics[doc_index][tok_index]
        vocab_beta = len(self.types_to_numbers)*self.beta
        
        total = 0
        for j in xrange(self.num_topics):
            try:
                #~ top_left = self.topics[j][token_number] + self.beta
                #~ bottom_left = self.topic_totals[j] + vocab_beta
                #~ top_right = document[CountMaster.DOCUMENT_TOPIC_COUNTS][j] + self.alpha
                #~ bottom_right = self.get_num_tokens(doc_index) + topic_alpha
                #~ if j == token_topic:
                    #~ bottom_left -= 1
                    #~ top_right -= 1
                #~ left = top_left/bottom_left
                #~ right = top_right/bottom_right
                #~ pi_j = left*right
                #~ pi.append(pi_j)
                #~ total += pi_j
                top_left = self.topics[j][token_number] + self.beta
                bottom_left = self.topic_totals[j] + vocab_beta
                top_right = self.document_topic_counts[doc_index][j] + self.alpha
                if j == token_topic:
                    bottom_left -= 1
                pi_j = (top_left*top_right)/bottom_left
                pi.append(pi_j)
                total += pi_j
            except:
                print()
                print(top_left)
                print(bottom_left)
                print(self.topics[j][0])
                print(self.alpha)
                print(self.beta)
                print(vocab_beta)
                print()
                raise
        
        return pi, total
        
    def select_new_topic(self, pi, total):
        r = random.random()*total
        total_num = 0
        for topic, num in enumerate(pi):
            total_num += num
            if r < total_num:
                break
        return topic
    
    def update_token_topic_assignment(self, doc_index, tok_index, new_topic):
        token_number = self.document_token_numbers[doc_index][tok_index]
        old_topic = self.document_token_topics[doc_index][tok_index]
        self.document_token_topics[doc_index][tok_index] = new_topic
        self.document_topic_counts[doc_index][old_topic] -= 1
        self.document_topic_counts[doc_index][new_topic] += 1
        self.topic_totals[old_topic] -= 1
        self.topics[old_topic][token_number] -= 1
        self.topic_totals[new_topic] += 1
        self.topics[new_topic][token_number] += 1
    
    def print_topic(self, topic, top_n):
        t = {}
        for token_number, count in enumerate(self.topics[topic]):
            t[token_number] = count
        s = sorted(t, cmp=lambda x,y: y-x, key=lambda x: t[x])
        s = s[0:top_n]
        print('Topic %s: '%(str(topic)), end='')
        for token_number in s:
            print('%s(%s) '%(str(self.numbers_to_types[token_number]), str(t[token_number])), end='')
    
    def log_likelihood(self):
        V = len(self.numbers_to_types)
        T = self.num_topics
        topics = self.topics
        B = self.beta
        VB = V*B
        
        ll = T*(gammaln(VB) - V*gammaln(B))
        
        for j in xrange(T):
            temp_topic_ll = -gammaln(self.topic_totals[j] + VB)
            for w_count in topics[j]:
                temp_topic_ll += gammaln(w_count + B)
            ll += temp_topic_ll
        
        return ll

class LDA(object):
    def __init__(self, num_topics, alpha=0.1, beta=0.1, burn=10, length=10):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.burn = burn
        self.length = length
        self.total_tokens = 0
        self.log_likelihoods = []
    
    def run_analysis(self, document_iterator, seed=0):
        random.seed(seed)
        
        print('----------Initializing----------')
        self.cm = cm = CountMaster(self.num_topics, self.alpha, self.beta)
        
        # preprocess documents
        for tokens in document_iterator:
            cm.add_tokens(tokens)
        
        cm.done_adding_tokens()
        
        self.log_likelihoods.append(cm.log_likelihood())
        
        print('----------Sampling----------')
        for time_step in xrange(self.burn+self.length):
            print('Time Step: %s'%(str(time_step)), end='')
            start_time = time.time()
            for doc_index in xrange(cm.get_num_documents()):
                for tok_index in xrange(cm.get_num_tokens(doc_index)):
                    # get pi vector
                    pi, total = cm.get_pi_vector(doc_index, tok_index)
                    #~ print(pi)
                    # select topic assignment
                    new_topic = cm.select_new_topic(pi, total)
                    # update counts
                    cm.update_token_topic_assignment(doc_index, tok_index, new_topic)
            log_likelihood = cm.log_likelihood()
            print('; Log Likelihood: %s'%(str(log_likelihood)), end='')
            self.log_likelihoods.append(log_likelihood)
            print('; Seconds: %s'%(str(time.time()-start_time)))
        
        print('----------Done Sampling-----------')
    
    def print_topics(self, top_n):
        for t in xrange(self.num_topics):
            self.cm.print_topic(t, top_n)
            print()
