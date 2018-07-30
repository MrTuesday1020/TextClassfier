import helper
import numpy as np
import operator
from sklearn.feature_extraction.text import TfidfTransformer

def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()
    class0 = strategy_instance.class0 
    class1 = strategy_instance.class1
    
    len_of_class0 = len(class0)
    len_of_class1 = len(class1)
    len_of_train = len_of_class0 + len_of_class1

    feature_set = set()
    for paragrah in class0:
        feature_set = set.union(feature_set, paragrah)
    for paragrah in class1:
        feature_set = set.union(feature_set, paragrah)
        
    word_list = list(feature_set)
    len_of_feature = len(word_list)

    class0_word_count = np.zeros((len_of_class0, len_of_feature), dtype = int)
    class1_word_count = np.zeros((len_of_class1, len_of_feature), dtype = int)

    y_train = np.zeros((len_of_train,), dtype = int)

    # count words
    for i in range(len_of_class0):
        for word in class0[i]:
            word_index = word_list.index(word)
            class0_word_count[i][word_index] += 1
                
    for i in range(len_of_class1):
        for word in class1[i]:
            word_index = word_list.index(word)
            class1_word_count[i][word_index] += 1
        y_train[i+len_of_class0] = 1

    x_train = np.append(class0_word_count, class1_word_count, axis=0)

    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(x_train)
    
    parameters={'gamma':0.01, 'C':1.0, 'kernel':'linear', 'degree':3, 'coef0':3.0}    
    clf = strategy_instance.train_svm(parameters, tfidf, y_train)
    
    # get weights of the classification
    weights_list = clf.coef_[0].toarray()[0]
    weights_dict = {}
    for i in range(len_of_feature):
        weights_dict[word_list[i]] = weights_list[i]
        
    # sort weights
    sorted_weights = sorted(weights_dict.items(), key = operator.itemgetter(1))
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    with open(test_data, 'r') as f:
        test = [line.strip().split(' ') for line in f]
    
    new_paragraphs = []
    for paragrah in test:
        temp_weights_dict = {}
        for word in paragrah:
            if word not in temp_weights_dict and word in weights_dict:
                this_weight = weights_dict[word]
                temp_weights_dict[word] = this_weight
        temp_weights_dict = sorted(temp_weights_dict.items(), key = operator.itemgetter(1),reverse=True)
        # delete 15 words
        for word in temp_weights_dict[:15]:
            paragrah[:] = [ i for i in paragrah if i != word[0] ]
        # add 5 words
        count = 0
        for word in sorted_weights:
            if word[0] not in paragrah and count != 5:
                paragrah.append(word[0])
                count += 1
        new_paragraph = ' '.join(paragrah)
        new_paragraph += '\n'
        new_paragraphs.append(new_paragraph)
  
    modified_data='./modified_data.txt'
    with open(modified_data, 'w') as f:
        f.writelines(new_paragraphs)
    
    
    ## You can check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.