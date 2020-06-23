import pprint
import random
from keras import *
#import matplotlib.pyplot as plt
import numpy as np
import math
from getData import *
import os
#import tensorflow as tf
#gpu_id = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
#os.system('echo $CUDA_VISIBLE_DEVICES')
#tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
#tf_config.gpu_options.allow_growth = True
#tf.Session(config=tf_config)
import gc
import pickle
#from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from keras.callbacks import  EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
#import pylab as pl
from keras.utils.generic_utils import get_custom_objects
from keras.backend import sigmoid
#random.seed(7)
#numpy.random.seed(7)
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': swish})

class Network():
    '''
    Class that represents the network to be evolved.
    '''
    def __init__(self):
        self.fitness = []
        self.finalAuc = 0
        self.layers = [("Dense", "tanh", 1, .25)]
        self.network = Sequential()
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'Adam'
        self.train_plot = None
        self.trained = False
        self.train_epochs = 30


    def randomize(self):
        '''
        Completely randomize the attributes of the network
        '''
        ##loss function###
        self.loss = "categorical_crossentropy"
        #random.choice(["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error",
        #                           "squared_hinge", "sparse_categorical_crossentropy", "kullback_leibler_divergence",
        #                           "cosine_proximity"])
        #optimizer: if it is not changed?
        self.optimizer = random.choice([optimizers.Adam(lr=1e-4)])
        #self.optimizer =  optimizers.Adam(lr=1e-4)
        #Generate layers: total depth
        #depth = random.randrange(1, 5)
        depth = 2
        self.layers = [None] * depth
        for l in range(depth):
            if l == 0:
                self.layers[l] = self.randomLayer(top=True)
            else:
                self.layers[l] = self.randomLayer()
        #self.train_epochs = random.randrange(50, 200)
        self.train_epochs = 30
    def build(self, input):
        '''
        Build the network - we take our list of layer attributes and use them to make actual layers in the model.
        '''
        ###code?
        self.network = Sequential()
        x,y,z = input.shape
        #input.shape = [None, x, y, z]
        #first depth: input_dim
        #input = input.reshape(y,z)
        # print(input.shape)
        newshape = (y, z)
        print(newshape)
        # print(newshape)
        # print(self.layers)
        #Input layer
        if self.layers[0][0] == "Dense":
            self.network.add(layers.Dense(self.layers[0][2], activation=self.layers[0][1], input_shape=newshape))
        if self.layers[0][0] == "Conv":
            self.network.add(layers.Conv1D(self.layers[0][2], 3, activation=self.layers[0][1], input_shape=newshape))
            #self.network.add(layers.BatchNormalization())
        if self.layers[0][0] == "Conv+Pool":
            print(newshape)
            #the initialization for the weights?
            self.network.add(layers.Conv1D(self.layers[0][2], 3, activation=self.layers[0][1], input_shape=newshape))
            #self.network.add(layers.Conv1D(input_dim=21, input_length=99, nb_filter=102, filter_length=7, border_mode="valid",
                          #activation="relu", subsample_length=1))
            self.network.add(layers.MaxPooling1D(pool_size=2))
            #none better?
            #self.network.add(layers.Dropout(0.5))
            #self.network.add(layers.BatchNormalization())
            #model.add(Dropout(0.5))
            #self.network.add(layers.AveragePooling1D(pool_size=5))
        if self.layers[0][0] == "LSTM":
            self.network.add(layers.LSTM(self.layers[0][2], activation=self.layers[0][1], return_sequences=True))
        for i in range(1, len(self.layers)):
            if self.layers[i][0] == "Dense":
                self.network.add(layers.Dense(self.layers[i][2], activation=self.layers[i][1]))
            if self.layers[i][0] == "Conv":
                self.network.add(layers.Conv1D(self.layers[i][2], 3, activation=self.layers[i][1]))
                #self.network.add(layers.BatchNormalization())
            #change????
            if self.layers[i][0] == "Conv+Pool":
                self.network.add(layers.Conv1D(self.layers[i][2], 3, activation=self.layers[i][1]))
                self.network.add(layers.MaxPooling1D(pool_size=2))
                #self.network.add(layers.BatchNormalization())
            if self.layers[i][0] == "LSTM":
                self.network.add(layers.Bidirectional(layers.LSTM(self.layers[i][2], activation=self.layers[i][1], return_sequences=True)))
                #self.network.add(layers.BatchNormalization())
            #if self.layers[i][0] == "GausNoise":              
                    #self.network.add(layers.GaussianNoise(self.layers[i][3]))
            #if self.layers[i][0] == "BatchNorm":
                    #self.network.add(layers.BatchNormalization())
                #self.network.add(layers.LSTM(128, input_dim=102, input_length=31, return_sequences=True))
            #if self.layers[i][0] == "Flatten":
                #self.network.add(layers.Flatten())
        # Dropout and GausNoise use self.layers[i][3]
        self.network.add(layers.Flatten())
        #self.network.add(layers.BatchNormalization())
        self.network.add(layers.Dropout(0.25))
        self.network.add(layers.Dense(2, activation='softmax'))
        #self.network.compile(loss=self.loss, optimizer=self.optimizer, metrics=['acc'])
        self.network.compile(loss=self.loss, optimizer=self.optimizer)
        #self.network.compile(loss=self.loss, optimizer=optimizers.Adam(lr=1e-4))
        #plot_model(self.network, to_file='model.png', show_shapes = True)
        return input


    def train(self, verbose=False, stop_for_errors=False, quicktrain=False):
        """
        Train the network and record the fitness score.
        """
        trainX, test_X, trainY, test_y = dealwithdata('FOX2')
        test_y = test_y[:,1]
        # print(trainX)
        # print(trainX.shape)
        # print(type(trainX))
        # print(trainY.shape)
        # print('*********')
        if verbose:
            print("Now attempting to build and train the following network: ")
            print("Input shape: " + str(trainX.shape))
            pprint.pprint(self.layers)

        #try: too slow? GPU tensorflow backend
        #should be 5
        kf = KFold(2, True) 
        #cross-validation
        #verbose = 1:show the progress bar
        aucs = []
        sens = []
        spes = []
        if quicktrain: epochs = 5
        else:
            epochs = self.train_epochs
        for train_index, eval_index in kf.split(trainY):
             earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
             train_X = trainX[train_index]
             train_y = trainY[train_index]
             eval_X = trainX[eval_index]
             eval_y = trainY[eval_index]
             train_X = self.build(train_X)
             history = self.network.fit(train_X, train_y, batch_size=50, epochs=epochs,validation_data=(eval_X, eval_y),
                   callbacks=[earlystopper])
             loss = history.history['loss']
             val_loss = history.history['val_loss']
             predictions = self.network.predict_proba(test_X)[:, 1]
             predict = self.network.predict_classes(test_X).astype('int')
             cm = confusion_matrix(test_y,predict)
             sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
             specificity = cm[0,0]/(cm[0,0]+cm[0,1])
             # print('Sensitivity : ', sensitivity)
             # print('Confusion Matrix : \n', cm)
             # print(predictions)
             #print(predictions)
             auc = roc_auc_score(test_y, predictions)
             aucs.append(auc)
             sens.append(sensitivity)
             spes.append(specificity)
             
        # for i in range(2):
        #     print(aucs[i])
        #how to calculate if epochs is not 30 (for the earlystopper)#       
        # acc = history.history['acc']
        # history_dict = history.history
        # print(history_dict.keys())
        # val_acc = history.history['val_acc']
        # epochs = range(len(loss))
        auc = np.mean(aucs)
        obj1 = np.mean(sens)
        obj2 = np.mean(spes)
        #minimization
        self.fitness = [-obj1,-obj2]
        self.finalAuc = auc
        #self.fitness = np.mean(aucs)
        print(self.fitness)
        print(self.finalAuc)
        print("*************")
        # if math.isnan(self.fitness):
        #     self.fitness = -1

        #Graph and output
        # plt.figure()
        # plt.plot(epochs, loss, 'bo', label='Training loss')
        # plt.plot(epochs, val_loss, 'b', label='Validation loss')
        # plt.title("Training and Validation Loss " +  " - Fitness: " + str(self.fitness))
        # plt.legend()
        # self.train_plot = plt
        print("====================== Netword added to population! ======================")
        print("Input shape: " + str(trainX.shape))
        pprint.pprint(self.layers)

        # except Exception as e:
        #     if stop_for_errors:
        #         raise e
        #     else:
        #         print("An error occurred while training")
        #         print(e.message)
        #         self.fitness = -1
        #         print("Network thrown out")
        self.trained = True

    def mutate(self):
        # if all the layers have pool and with hypermutate, then the maximum depth is 2 (99/2) 
        '''
        how to mutate
        Mutates some aspect of the network at random. Must be built afterwards.
        '''
        type = random.randrange(6)
        #if type == 1:
            #self.loss = "categorical_crossentropy"
        #random.choice(["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error",
        #                           "squared_hinge", "sparse_categorical_crossentropy", "kullback_leibler_divergence",
        #                           "cosine_proximity"])
        if type == 1:
            self.optimizer = random.choice(
                [optimizers.Adam(), optimizers.SGD(), optimizers.RMSprop()])
            #self.optimizer = optimizers.Adam(lr=1e-4)
        elif type == 2:
            if len(self.layers) >= 2:
                self.layers.pop(random.randrange(len(self.layers)))
            else:
                #insert(index,object) insert object before index
                self.layers.insert(random.randrange(0, len(self.layers)), self.randomLayer())
        elif type == 3:
            self.layers.insert(random.randrange(0, len(self.layers)), self.randomLayer())
        elif type == 4:
            self.layers[random.randrange(0, len(self.layers))] = self.randomLayer()
        else: 
            #type == 5:
            self.layers[0] = self.randomLayer(top=True)
        # else:
        #     l = random.randrange(0, len(self.layers))
        #     self.layers[l] = self.randomLayer()

    def hypermutate(self, repetitions = 3):
        '''
        Mutates multiple times
        '''
        for i in range(repetitions):
            self.mutate()

    def randomLayer(self, top=False):
        '''
        Creates a randomly generated layer four parameters
        '''
        # the network is not stable?  the initialization network?
        layer = [None, None, None, None]
        #no Dense?
        if top:
        #the first layer
           #layer[0] = random.choice(["Dense", "Conv", "Conv+Pool", "LSTM"])
           layer[0] = random.choice(["Conv+Pool"])
           #layer[0] = random.choice(["Conv", "Conv+Pool"])
        else:
            #layer[0] = random.choice(
                #["Dense", "Conv", "LSTM", "GausNoise", "Dropout", "BatchNorm"])
            layer[0] = random.choice(
                ["LSTM"])
            #layer[0] = random.choice(
                #["Conv", "Conv+Pool", "LSTM"])
        #layer[1] = random.choice(["relu", "tanh"])
        #????
        layer[1] = random.choice(["relu"])
        #layer[1] = random.choice(["swish"])
        #layer[1] = random.choice(["sigmoid", "relu", "tanh", "softmax"])
        layer[2] = random.randrange(64, 257)
        #layer[2] = 102
        ##real hyper-parameter 0-0.99
        layer[3] = random.randrange(99)/100
        
        return tuple(layer)

    # def print_plot(self):
    #     if self.train_plot != None:
    #         self.train_plot.savefig('1.png', bbox_inches='tight')


class Population():
    '''
        Represents a population of networks which can breed together and evolve
    '''
    def __init__(self, count, population_file=None):
        self.pop = []
        self.net_retain_pct = .4
        self.reject_survival_pct = .1
        # (Out of 10)
        self.mutate_chance = 6
        self.hypermutate_chance = 9
        while len(self.pop) < count:
            # pop is n Network class #
            n = Network()
            n.randomize()

            trainable_layers = 0
            for layer in n.layers:
                if layer[0] == "Dense" or layer[0] == "Conv" or layer[0] == "Conv+Pool" or layer[0] == "LSTM":
                    trainable_layers += 1
            if trainable_layers == 0:
                del n.network
                del n
                gc.collect()
                continue

            n.train(quicktrain=False)
            if n.fitness != -1:
                self.pop.append(n)
            else:
                del n.network
                del n
                gc.collect()

    # def save_pop(self):
    #     pop = tuple(self.pop)
    #     for net in pop:
    #         net.trained = False
    #         net.network = None
    #         net.train_plot = None
    #     file = open("pop.obj", "wb")
    #     pickle.dump(pop, file, pickle.HIGHEST_PROTOCOL)

    # def load_pop(self):
    #     file = open("pop.obj", "wb")
    #     pop = pickle.load(file)
    #     for net in pop:
    #         net.trained = False
    #         net.network = Sequential()
    #         net.train()
    #     self.pop = list(pop)
    
    
    
    #crossover#
    def crossover(self, mother, father):
        '''
        Take two nets and create combined (and sometimes mutated) "children" which are combinations of them.
        '''
        children = []
        for i in range(2):

            child = Network()
            child.layers = random.choice([mother.layers, father.layers])
            child.loss = random.choice([mother.loss, father.loss])
            child.optimizer = random.choice([mother.optimizer, father.optimizer])
            #Randomly mutate some children
            if random.randrange(10) > self.mutate_chance:
                child.mutate()
            elif random.randrange(10) > self.hypermutate_chance:
                child.hypermutate()

            children.append(child)
        return children
    
    #Minimization
    def Dominate(self,x,y):
        for i in range(len(x.fitness)):
            if x.fitness[i] > y.fitness[i]:
                return False
        return True
    
    def Neighbor(self,Lamb,T):
    #Lambda list,numbers of neighbors is T
        B=[]
        for i in range(len(Lamb)):
            temp=[]
            for j in range(len(Lamb)):
                distance=np.sqrt((Lamb[i][0]-Lamb[j][0])**2+(Lamb[i][1]-Lamb[j][1])**2)
                temp.append(distance)
            l=np.argsort(temp)
            B.append(l[:T])
        return B

    def BestValue(self):
    #get the bestvalues of each function,which used as the reference point
    #if goal for function is minimazaton,z is the minimize values
    #P:population
        P = self.pop
        best=[]
        for i in range(len(P[0].fitness)):
            best.append(P[0].fitness[i])
        for i in range(1,len(P)):
            for j in range(len(P[i].fitness)):
                if P[i].fitness[j]<best[j]:
                    best[j]=P[i].fitness[j]
        return best
    
    def Tchebycheff(self,x,lamb,z):
    #Tchebycheff approach operator

        temp=[]
        for i in range(len(x.fitness)):
            temp.append(np.abs(x.fitness[i]-z[i])*lamb[i])
        return np.max(temp)
    
    def evolve_MOEA(self, N, T):
        
        #initialize the weight for two objectives#
        Lamb=[]
        for i in range(N):
            temp=[]
            temp.append(float(i)/(N))
            temp.append(1.0-float(i)/(N))
            Lamb.append(temp)
            
        for network in self.pop:
            if not network.trained:
               network.train()
        
        p = self.pop
        
        B = self.Neighbor(Lamb,T)
        
        z = self.BestValue()
        
        EP=[]
        
        #the number of iterations 
        t=0
        
        MaxIter = 1
        
        while(t<MaxIter):
            t+=1
            #EP = 1
            print('PF number:',len(EP))
            #step into the loop
            for i in range(N):

                k = random.randint(0, T - 1)
                l = random.randint(0, T - 1)
                
                y1,y2 = self.crossover(p[B[i][k]], p[B[i][l]])
                
                print(y1.layers)
                print(y2.layers)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                y1.train()
                y2.train()
                
                if self.Dominate(y1,y2):
                    y=y1
                else:
                    y=y2
                
                for j in range(len(z)):
                    if y.fitness[j] < z[j]:
                        z[j] = y.fitness[j]
                for j in range(len(B[i])):
                    Ta = self.Tchebycheff(p[B[i][j]], Lamb[B[i][j]], z)
                    Tb = self.Tchebycheff(y, Lamb[B[i][j]], z)
                    if Tb < Ta:
                        p[B[i][j]] = y
                if EP == []:
                    EP.append(y)
                else:
                    dominateY = False
                    rmlist=[]
                    for j in range(len(EP)):
                        if self.Dominate(y, EP[j]):
                            rmlist.append(EP[j])
                        elif self.Dominate(EP[j], y):
                            dominateY = True
    
                    if dominateY == False:
                        EP.append(y)
                        for j in range(len(rmlist)):
                            EP.remove(rmlist[j])
    
        x = []
        y = []
        for i in range(len(EP)):
            x.append(EP[i].fitness[0])
            y.append(EP[i].fitness[1])
            print(EP[i].finalAuc)
        plt.plot(x, y, '*')
        plt.xlabel('sensitivity')
        plt.ylabel('specificity')
        plt.show()
        for i in range(len(p)):
            print(p[i].finalAuc)
        

if __name__ == "__main__":
    N = 2
    p = Population(N)
    #p.save_pop()
    #the number of neighbors
    T = 2
    p.evolve_MOEA(N, T)
    # graded = [(network.fitness, network) for network in lastPop]     
    # graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
    # print("Last result " + str(graded[0].fitness))
        
    # f = open('result.txt','w')
    # f.write('The last AUC: %.4f' %(graded[0].fitness))
    # f.close()
    