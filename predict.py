import argparse
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model.resnet import ResNet50
from data.data import load_set

import tensorflow as tf
import numpy as np
import logging

from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


from PIL import Image
acc=[]
y_test=[]
y_pred=[]


JSON_CONFIG = 'config.json'




def show_results(filename, classname, accuracy):
    image = Image.open(filename)
    plt.imshow(image)
    plt.title("This is {} with {}%".format(classname, accuracy))
    #plt.title("This is {} with {}%  filename {}".format(classname, accuracy, filename))
    plt.show()


def interpret(filenames, predictions, classes_dict):
    assert len(filenames) == predictions.shape[0]

    csv_dir="/home/rkaratt/TeamProject/resnet50-tensorflow/GT-final_test.csv"
    #df = pd.read_csv(csv_dir , sep=';', header=0, index_col =2)
    df = pd.read_csv(csv_dir , sep='\t', index_col=0)
    #print(df)

    for i, file in enumerate(filenames):
        print(file)
        pred_path=Path(file)
        name_pred=pred_path.name
        #name=str(name_pred)
        #print(name)
        test_label=df.loc[name_pred ,'ClassId']
        #test_label=df.loc[name_pred,'ClassId']

        #print(name_pred)
        #print(test_label)
        test_label=int(test_label)
        #print(type(test_label))
        prediction = predictions[i]
        class_index = np.argmax(prediction)
        accuracy = prediction[class_index]
        class_name = classes_dict[class_index]
        class_name = int(class_name)
        #print(type(class_name))
        print("is {} with {}%".format(class_name, accuracy * 100))
        #print("is {} with {}%  tclass {} ".format(class_name, accuracy * 100, filenames,i))
        #show_results(file, class_name, accuracy * 100)
        y_test.append(test_label)
        y_pred.append(class_name)
        acc.append(accuracy)

    #print(type(y_test))
    #print(type(y_pred))
    print(y_test)
    print(y_pred)
    plt.plot(acc,'ro')
    plt.ylabel('Accuracy')
    plt.xlabel('Photo')
    plt.show()
    #test_accuracy = (sum(y_test == y_pred))/len(y_test)
    #print(sum(i[0] == i[1] for i in zip(y_test, y_pred)))
    #test_accuracy = (sum(set(y_test) & set(y_pred)))/len(y_test)
    test_accuracy = (sum(i[0] == i[1] for i in zip(y_test, y_pred)))/len(y_test)
    #print(set(y_test) & set(y_pred))
    print(test_accuracy)
    #average = sum(acc) / len(acc)  
    #print(average)  

    cm = confusion_matrix(y_test, y_pred)
    cm1 = cm.astype('float')/(cm.sum(axis=0)[:, np.newaxis]) 
    cm2 = np.log(.0001 + cm1)
    plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Log of normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #TP = tf.count_nonzero(y_pred * y_test)








    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))


    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5', 'Class 6', 'Class 7','Class 8', 'Class 9', 'Class 10', 'Class 11', 'Class 12', 'Class 13', 'Class 14', 'Class 15', 'Class 16', 'Class 17', 'Class 18', 'Class 19', 'Class 20', 'Class 21', 'Class 22', 'Class 23', 'Class 24', 'Class 25', 'Class 26', 'Class 27', 'Class 28', 'Class 29', 'Class 30', 'Class 31','Class 32', 'Class 33', 'Class 34', 'Class 35', 'Class 36', 'Class 37', 'Class 38', 'Class 39', 'Class 40', 'Class 41', 'Class 42', 'Class 43']))
     

    
    




  


def predict(model_folder, image_folder, classes_dict, debug=False):
    weights = os.path.join(model_folder, 'model.ckpt')
    n_classes = len(classes_dict)
    model = ResNet50(JSON_CONFIG, n_classes)
    filenames = model.load_pred(image_folder)
    predictions = model.predict(weights, debug=debug)
    interpret(filenames, predictions, classes_dict)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-img", "--img-folder", required=True,
                        help="specify path to images to make prediction")
    parser.add_argument("-f", "--data-folder", required=True,
                        help="path to Training Dataset to get class dict")
    parser.add_argument("-mod","--model-folder", required=True,
                        help="specify path to folder with saved model")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Use TensorFlow Debugger")

    args = parser.parse_args()

    model_folder = args.model_folder
    image_folder = args.img_folder
    dataset_folder = args.data_folder
    debug = args.debug

    classes_dict = load_set(dataset_folder, only_dict=True)
    logging.info(classes_dict)

    predict(model_folder, image_folder, classes_dict, debug)


if __name__ == '__main__':
    main()
