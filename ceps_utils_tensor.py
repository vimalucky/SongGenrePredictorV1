import timeit
import glob
import os
import numpy as np
import scipy.io.wavfile
import sklearn
import sys
from scikits.talkbox.features import mfcc
from config import TRAIN_DATASET_DIR,TEST_DATASET_DIR,GENRE_LIST

genre_list={}

def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    ceps = np.load(data_fn+".npy")
    num_ceps = len(ceps)
    print (num_ceps)


def create_train_ceps():
    """
        Creates the MFCC features from the train files,
        saves them to disk, and returns the saved file name.
    """
       
    for subdir, dirs, files in os.walk(TRAIN_DATASET_DIR):
        genre = subdir[subdir.rfind('/',0)+1:]
        print (genre)
        if genre in genre_list:
            count=0
            genre_ceps=np.zeros((70,13),dtype=float)
            print (subdir)
            for file in files:
                path = subdir+'/'+file
                #print path
                if path.endswith("wav"):
                    #print path
                    #create_ceps(path)
                    sample_rate, X = scipy.io.wavfile.read(path)
                    ceps, mspec, spec = mfcc(X)
                    num_ceps = len(ceps)
                    #print(num_ceps)
                    #sys.exit(1)
                    ceps=np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
                    # ceps = ceps[2000:2010,:]
                    # ceps = ceps.ravel()
                    #print len(ceps)
                    #sys.exit(1)
                    genre_ceps[count]=ceps
                    count=count+1
                    
            print (count)
            #genre_ceps = np.array(genre_ceps)
            print (genre_ceps.shape)
            #break
            write_ceps(genre_ceps,path) 


def read_train_ceps(genre_list, base_dir=TRAIN_DATASET_DIR):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            #num_ceps = len(ceps)
            #X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
            for feature in ceps:
                X.append(feature)
                y.append(label)
    return np.array(X), np.array(y)

def create_test_ceps():
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """    
    for subdir, dirs, files in os.walk(TEST_DATASET_DIR):
        genre = subdir[subdir.rfind('/',0)+1:]
        print (genre)
        if genre in genre_list:
            count=0
            genre_ceps=np.zeros((30,13),dtype=float)
            print (subdir)
            for file in files:
                path = subdir+'/'+file
                #print path
                if path.endswith("wav"):
                    #print path
                    #create_ceps(path)
                    sample_rate, X = scipy.io.wavfile.read(path)
                    ceps, mspec, spec = mfcc(X)
                    num_ceps = len(ceps)
                    ceps=np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
                    # ceps = ceps[2000:2010,:]
                    # ceps = ceps.ravel()
                    genre_ceps[count]=ceps
                    count=count+1
                    
            print (count)
            #genre_ceps = np.array(genre_ceps)
            print (genre_ceps.shape)
            #break
            write_ceps(genre_ceps,path)

def read_test_ceps(genre_list, base_dir=TEST_DATASET_DIR):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            #num_ceps = len(ceps)
            #X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
            for feature in ceps:
                X.append(feature)
                y.append(label)
    return np.array(X), np.array(y)

if __name__== "__main__" :
	start_time = timeit.default_timer()
	for subdir,dirs,files in os.walk(TRAIN_DATASET_DIR):
		genre_list = list(set(GENRE_LIST).intersection(set(dirs)))
		break
	print ("Working with these genres --> ", genre_list)
	print ("Starting ceps generation")
	create_train_ceps()
	create_test_ceps()
	stop_time = timeit.default_timer()
	print ("Total ceps generation and feature writing time (s) = ",(stop_time-start_time))
    
