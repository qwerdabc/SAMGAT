import random
from random import shuffle
import os

cwd=os.getcwd()




def load5foldData(obj,seed): 
    random.seed(seed)
    if obj == "Phemeold":
        rumor_path = './data/phemeold/rumours/'
        non_rumor_path = './data/phemeold/non-rumours/'
        
        rumor_dirs = os.listdir(rumor_path)
        non_rumor_dirs = os.listdir(non_rumor_path)


        #l1,l2 = len(rumor_dirs), len(non_rumor_dirs)

        F = rumor_dirs
        T = non_rumor_dirs

        #T = T[0:1789]

        #print('rumor : non-romor', len(F), len(T))
        random.shuffle(F) 
        random.shuffle(T) 

        l1,l2 = len(F), len(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)

        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        #print(len(F),len(T))

        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        #print(len(F),len(T))

        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        #print(len(fold2_x_test))
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        #print(len(fold2_x_test))

        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        #print(len(F),len(T))

        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])
    if obj == "Pheme":
        rumor_path = './data/pheme/rumor/'
        non_rumor_path = './data/pheme/non-rumor/'
        
        rumor_dirs = os.listdir(rumor_path)
        non_rumor_dirs = os.listdir(non_rumor_path)


        #l1,l2 = len(rumor_dirs), len(non_rumor_dirs)

        F = rumor_dirs
        T = non_rumor_dirs

        #T = T[0:1789]

        #print('rumor : non-romor', len(F), len(T))
        random.shuffle(F) 
        random.shuffle(T) 

        l1,l2 = len(F), len(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)

        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        #print(len(F),len(T))

        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        #print(len(F),len(T))

        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        #print(len(fold2_x_test))
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        #print(len(fold2_x_test))

        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        #print(len(F),len(T))

        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

    fold0_test = list(fold0_x_test) 
    shuffle(fold0_test) 
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    shuffle(fold4_train)

    return list(fold0_test),list(fold0_train),\
           list(fold1_test),list(fold1_train),\
           list(fold2_test),list(fold2_train),\
           list(fold3_test),list(fold3_train),\
           list(fold4_test), list(fold4_train)
