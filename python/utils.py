from sklearn.cross_validation import KFold

def generate_kfold(rows,n_folds=5,shuffle=True,random_state=None):
    #rows = [row for row in csv_reader]
    
    n = len(rows)
    kf = KFold(n, n_folds=n_folds, shuffle=shuffle,random_state=random_state)
    
    folds = []
    for train_index, test_index in kf:
        folds += [([rows[i] for i in train_index],[rows[i] for i in test_index])]

    return folds

if __name__ == "__main__":
    ret = generate_kfold(range(100),n_folds=5,shuffle=True,random_state=1634120)
    
    print len(ret)
