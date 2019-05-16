__authur__ = "tangxi.zq"
__time__ = "2019-05-14"

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage
from skimage.io import imread
from skimage.transform import resize
from scipy import ndimage
from skimage.feature import local_binary_pattern
import cv2
import joblib
import os


def get_images(path):
    return [os.join(path,each) for each in os.lisdir(path) if each.endswith('.png') or each.endswith('.jpg')]

radius = 3
def filter_median(img, factor=2):
    return ndimage.median_filter(img, factor)
numPoints = 8*radius
def localBinaryPatterns(img, res, numPoints=24, radius=2):
    
    img_lbp = np.zeros(np.array(img).shape)
    for i in range(np.array(img).shape[2]):
        lbp = local_binary_pattern(img[:,:,i], numPoints,
                    radius, method="uniform")
        
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, radius + 2))
    
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        res.append(hist.ravel())
    return list(np.array(res).reshape(3*26)), lbp

def get_image_feature(img):
    noise = np.abs(np.array(img).astype(float) - filter_median(img).astype(float))
    _,img_lbp = localBinaryPatterns(noise,[])
    img_lbp = cv2.resize(img_lbp,(256,256))
    return img_lbp
    
def load_image_files(container_path, dimension=(64, 64)):
    """
    sklearn 自带的载入数据方法
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    print(categories)
    descr = "dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            print(file)
            img = skimage.io.imread(file)
            img = get_image_feature(img)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
                 
def train():
    image_dataset = load_image_files("images/")
    X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)
    param_grid = [
                  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                 ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid)
    clf.fit(X_train, y_train)
    joblib.dump(clf,'save/clf.pkl',compress = 3)
    y_pred = clf.predict(X_test)
    print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))
    
def test():
    clf3 = joblib.load("save/clf.pkl")
    test_images = get_images("test_images/")
    for each in test_images:
        img = skimage.io.imread(each)
        img = get_image_feature(img)
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten()) 
        try:
            print(clf3.predict(img_resized.flatten()))
        except Exception as e:
            import logging
            logging.Exception(e)
    flat_data = np.array(flat_data)
    print(clf3.predict(flat_data[0:5]))

def main():
    train()
    test()
    print("done")
    
if __name__== '__main__':
    main()