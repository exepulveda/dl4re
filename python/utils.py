import numpy as np
from sklearn.model_selection import KFold
from keras.models import model_from_json
from keras.utils import to_categorical

def generate_kfold(rows,n_folds=5,shuffle=True,random_state=None):
    #rows = [row for row in csv_reader]
    
    n = len(rows)
    kf = KFold(n_splits=n_folds, shuffle=shuffle,random_state=random_state)
    
    folds = []
    for train_index, test_index in kf.split(rows):
        folds += [(rows[train_index],rows[test_index])]

    return folds

def save_model(model,model_filename):
    json_string = model.to_json()
    open(model_filename + ".json", 'w').write(json_string)
    model.save_weights(model_filename + ".h5",overwrite=True)

def load_model(model_filename):
    model = model_from_json(open(model_filename + ".json").read())    
    model.load_weights(model_filename + ".h5")

    return model
    
 
def encode_categories(y):
    cats = list(set(y))

    #code from 0 to ncats-1
    ycat = np.empty(len(y),dtype=int)
    code_dict = {}
    for i,c in enumerate(cats):
     code_dict[c] = i
     
     idx = np.where(y==c)[0]
     
     ycat[idx] = i
     

    return to_categorical(ycat),cats


if __name__ == "__main__":
    ret = generate_kfold(range(100),n_folds=5,shuffle=True,random_state=1634120)
    
    print(len(ret))
