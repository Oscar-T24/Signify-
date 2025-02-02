import random as r
from math import exp,sqrt
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
r.seed(1000)

#Useful constants
LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 0.001
DECAY = 0.995  # Reduce by 0.5% per sample
TESTFILE = "46inputs_bis.csv"

#   #   #   #   #

class Neuron :
    def __init__(self,parentnb,memorynb):
        """initialize the Neuron, by giving it random weight and bias """

        # weights
        scale = sqrt(2.0 / parentnb) if parentnb > 0 else 0.1
        self.weights = [r.uniform(-scale, scale) for _ in range(parentnb)]
        self.mw = sqrt(2.0 / parentnb) if parentnb > 0 else 0.1
        self.memory = 0        # memorized value
        self.bias = 0             #create a random bias to add on top of weights
        self.memiter = 0
        # forward pass
        self.z = 0
        self.a = 0
        
    
        # backpropagation
        self.error = 0      #error value used to compute gradient
        self.gw = []        #gradient of weights
        self.gmw = 0        #gradient of memory weights
        self.gb = 0         #gradient of bias
        self.data = {'Gw':[],'Gb':[],'Gmw': []}      #dic in which gradients will be saved, to compute the average and substract it to the weight and bias

    
    def getz(self,parentlayer):
        """compute the z on the node from its weights, bias and parentlayer , AND MEMORY VALUES"""
        self.z = 0 #reset z
        for i in range(parentlayer.nb):
            self.z += self.weights[i] * parentlayer.nodes[i].a
        self.z += self.bias
        if parentlayer.childlayer.childlayer != None :
            self.z += self.memory * self.mw
        return self.z
    
    def geta(self,function,allx = None):
        """compute a based on z and activation function"""
        if allx != None :
            self.a = function(self.z,allx)
            return function(self.z,allx)
        self.a = function(self.z)
        self.memory = self.a
        return function(self.z)

    
    def addMemory(self):
        """Circular buffer implementation"""
        if self.memory:  # Only if memory exists
            self.memiter = (self.memiter + 1) % len(self.memory)
            self.memory[self.memiter] = self.a

            
    def seta(self,values):
        """set a as a specific value"""
        self.a = values
        
    def update(self):
        """update weights and bias based on gradient"""
        max_grad_norm = 1.0  # Threshold for clipping

        # Clip bias gradient
        if abs(self.gb) > max_grad_norm:
            self.gb = max_grad_norm if self.gb > 0 else -max_grad_norm

        # Clip weight gradients
        for i in range(len(self.weights)):
            if abs(self.data['Gw'][-1][i]) > max_grad_norm:
                self.data['Gw'][-1][i] = (
                    max_grad_norm if self.data['Gw'][-1][i] > 0 
                     else -max_grad_norm
                )

        if abs(self.gb) > max_grad_norm:
            self.gmw = max_grad_norm if self.gmw > 0 else -max_grad_norm


            
        # Update weights and biases
        self.bias -= LEARNING_RATE * mean(self.data['Gb'])
        for i in range(len(self.weights)):
            self.weights[i] -= LEARNING_RATE * nestedmean(self.data['Gw'], i)
    
        self.mw -= LEARNING_RATE * mean(self.data['Gmw'])

        # Reset gradients
        self.data = {'Gw': [], 'Gb': [], 'Gmw': []}



class Layer :

    def __init__(self,childlayer,parentlayer,neuronNb,afunction,aderivative):
        """ afunction : activation function,
            aderivative : derivative of activation function"""
        self.childlayer = childlayer    
        self.parentlayer = parentlayer
        self.nodes = [] # list of nodes at that layer
        self.nb = neuronNb    # number of neurons on that layer
        self.afunction = afunction
        self.aderivative = aderivative

    ########### GETTERS ##########
    def isHidden(self):
        """if the layer is hidden or not"""
        return (self.parentlayer != None and self.childlayer != None)

    def getAllParams(self):
        """ return a nested list of all weights and bias of all neurons inside the array"""
        arr = []
        for node in self.nodes:
            arr.append(node.weights+[node.bias] + node.mw)    # list of weights + bias + memory weights at the end
        return arr
        # To improve : save data in tuples or nested list ?
    
    def getNodea(self):
        """get a of all nodes inside of the layer"""
        arr = []
        for node in self.nodes:
            arr.append(node.a)
        return arr

    def getNodez(self):
        """get z of all nodes inside of the layer"""
        arr = []
        for node in self.nodes:
            arr.append(node.z)
        return arr

    ################################

    
    def setNodea(self,values):
        """set a of all nodes inside of the layer to the values list"""
        if len(values) != self.nb :
            raise Exception("Invalid length")
        for i in range(len(values)):
            self.nodes[i].seta(values[i])
    
    def updateNodes(self,memorynb):
        """set the list of nodes of the layer.
            To use after initialization of the layer"""
        self.nodes = []
        if self.parentlayer == None :   # if it is the input layer
            for i in range(self.nb):
                self.nodes.append(Neuron(0,0)) # put no weights : not needed as it is the input layer
        else:
            for i in range(self.nb):     # else add new nodes
                self.nodes.append(Neuron(self.parentlayer.nb,memorynb))

    def setAlla(self,values = []):
        """compute a for all layers
            FORWARD PASS
            Return output layer"""
        if self.parentlayer == None :
            self.setNodea(values)
        else :
            for node in self.nodes:
                node.getz(self.parentlayer)
                if self.afunction == softmax :
                    node.geta(self.afunction,allx = self.getNodez())
                else :
                    node.geta(self.afunction)
                #print(node.a)
        if self.childlayer == None :
            return self
        else :
            return self.childlayer.setAlla()

    def delAllMemory(self):
        for node in self.nodes :
            node.memory = 0
        if self.childlayer != None :
            self.childlayer.delAllMemory()


    def setAllGradient(self,yArr = []):
        """get Gradient of layer.
            BACK PROPAGATION"""
        #print(self.nodes[0].mw)
        for i in range(self.nb):    # repeat for all nodes of layer
            currnode = self.nodes[i]
            currnode.gw = []        #clear weight gradient list
            currnode.gmw = 0

            ##### compute error of node #####
            if self.childlayer == None: # if it is the output layer
                if self.afunction == softmax : 
                    currnode.error  = (currnode.a - yArr[i])#*self.aderivative(currnode.z,self.getNodez()) #2(a - y) * dsigmoid
                else : 
                    currnode.error  = 2 * (currnode.a - yArr[i])*self.aderivative(currnode.z,self.getNodez()) #2(a - y) * dsigmoid
            else :
                currnode.error = 0
                for node in self.childlayer.nodes :
                    currnode.error += (node.error * node.weights[i]) 
                    
                currnode.error *= self.aderivative(currnode.z)
            #################################
                
            currnode.gb = currnode.error    # set the gradient of bias on all nodes of the layer

            for j in range(self.parentlayer.nb): # create the list of gradient of w on all nodes of the layer
                currnode.gw.append(self.parentlayer.nodes[j].a * currnode.error)
                
            currnode.gmw= currnode.memory * currnode.error # compute Gradient for memory weight
            
            currnode.data['Gw'].append(deepcopy(currnode.gw))
            currnode.data['Gb'].append(currnode.gb)
            currnode.data['Gmw'].append(currnode.gmw)
            
            if self.parentlayer.parentlayer != None :
                self.parentlayer.setAllGradient() # recurrence, repeat the process to the parent layer



    def updatewb(self):
        """update the weight and bias of all nodes on the layer and its child layers"""
        for node in self.nodes :
            node.update()       # update weight for all nodes
        if self.childlayer != None :
            self.childlayer.updatewb()  # recurrence : do it for next layer ( if not output layer)





class RNN:

    def __init__(self,neuronlist, hidden = "ReLU", output = "Softmax"):
        """initialize a Recursive Neural Network
            neuronlist : list of integers, representing each layer and the nb of neurons it has
            memory : nb of memory in a neuron"""
        self.head = None    # input layer
        self.tail = None    # output layer
        self.size = 0       # nb of layers
        #self.memory = memory # memory : nb of inputs in a sequence 
        for nb in neuronlist :
            self.add(Layer(None,None,nb,ACTIVATION[hidden][0],ACTIVATION[hidden][1]))
        self.tail.updateNodes(0)
        self.tail.afunction = ACTIVATION[output][0]
        self.tail.aderivative = ACTIVATION[output][1]

        
    def add(self,l):
        """add a layer to the Neural network, with memory"""
        if self.size == 0 :
            l.updateNodes(0)
            self.head = l
            self.tail = self.head
        else :
            l.parentlayer = self.tail
            l.updateNodes(1)
            self.tail.childlayer = l
            self.tail = l
        self.size +=1



    ######### Training ##########
    def train(self,datafile,samplesize,samplenb,randomize = True):
        """train the NN with the given datafile, where samplesize is the batch size.
            datafile (str) : path to data
            samplesize (int) : size of each sample (0 if the whole dataset is a single sample)
            samplenb (int) : number of sample """
        global LEARNING_RATE
        data = processData(datafile)    # process the csv file to make it usable
        cost = []
        for i in range( samplenb):      # repeat for the desired amount of samples
            sampleToUse = sample(samplesize,data)   # create random samples of desired size
            
            cost.append(self.sampleTrain(sampleToUse))
            LEARNING_RATE = max(MIN_LEARNING_RATE,LEARNING_RATE * DECAY)
        plt.plot( [i for i in range(len(cost))],cost)
        plt.show()
    def sampleTrain(self,sample):
        """ train the NN based on the sample, and output expected/actual value, as well as average cost"""
        cost = 0
        
        for case in sample:
            self.head.delAllMemory()
            expectedOutput = case.output.index(1)
            #print(case)
            for frame in case.data : 
                outputlayer = self.head.setAlla(frame)    # compute output
                outputa = self.tail.getNodea()
                actualOutput = outputa.index(max(outputa))  # find biggest output, and get its index -> actual number output
                for i in range(self.tail.nb):
                    cost+= ((self.tail.nodes[i].a - case.output[i])**2 )/len(case.data)  # cost function : sum of square of difference of results
                outputlayer.setAllGradient(case.output)         # backpropagate
            #print("expected : " + str(expectedOutput)+" / actual : "+ str(actualOutput) + " / certitude :" + str(max(outputa)))
        self.head.updatewb()                            # update weights and bias after training
        print("average cost : "+ str(cost/(len(sample))))
        return cost/len(sample)
    ##############################


    ##### Save Data #####    
    def save(self,SAVEFILE):
        """saves all weights and biases of the network on a pkl file"""
        curr = self.head
        savelist = []
        while curr != None:
            savelist.append(curr.getAllParams())
            curr = curr.childlayer
        with open(SAVEFILE,"wb") as file :
            pickle.dump(savelist,file)
    ######################

            
    def test(self,inputlist):
        """set the input to inputlist, and returns the output"""
        self.head.setAlla(inputlist)
        outputlist = self.tail.getNodea()
        output = outputlist.index(max(outputlist))
        return output


# Data classes
class Example:
    def __init__(self,data,output):
        self.data = data        # nested list
        self.output = toOutputList(output)
        self.length = len(data)
######## Helper Functions ########
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x,allx = None):
    
    return sigmoid(x)*(1-sigmoid(x))
def softmax(x, allx):
    exp_x = [exp(xi) for xi in allx]
    sum_exp = sum(exp_x)
    return exp(x) / sum_exp
def dsoftmax(x,allx):
    return softmax(x,allx)*(1-softmax(x,allx))
    #return softmax(x,allx)*(1-softmax(x,allx))
def ReLU(x):
    if x<0 :
        return 0.01*x
    else :
        return x
def dReLU(x) :
    if x<0 :
        return 0.01
    else :
        return 1
def deepcopy(l):
    """create a deepcopy of a list"""
    return [ i for i in l]
def mean(l):
    """return the average of a list"""
    if len(l) ==0:
        return 0
    return sum(l)/len(l)
def nestedmean(l,i):
    """return mean of ith element of each list in nested list"""
    if len(l) == 0:
        return 0
    return sum([nl[i] for nl in l])/len(l)


##functions to process data
def toOutputList(i):
    """turn i into a list where only the ith element is 1"""
    l = [0,0,0,0,0,0,0,0,0,0]
    l[i] = 1
    return l

def toInputArray(s) :
    """convert a string into an array"""
    arr = s.split()
    arr = [float(el) for el in arr]
    return arr


#### need to modify it for nested lists and format
def processData(FILE):
    """open FILE and process its data. """
    df = pd.read_csv(FILE)
    vidKeys = set(df[df.columns[0]])
    videos_raw = []
    dataset = []
    for key in vidKeys :
        raw_vid = df[(df[df.columns[0]] == key)]
        video_frames = []
        for i in range(len(raw_vid)):
            video_frames.append(list(raw_vid.iloc[i][1:-1]))
        dataset.append(Example(video_frames,int(raw_vid.iloc[0][-1])))
        
    
    return dataset


### Need to modify it for nested list
def sample(size,dataset):
    """take a sample of specified size of a larger dataset"""
    smpl = []
    for i in range(size):
        smpl.append(dataset[r.randrange(len(dataset))])
    return smpl

ACTIVATION = {"ReLU" : (ReLU,dReLU),"Sigmoid": (sigmoid, dsigmoid), "Softmax" : (softmax,dsoftmax)}


# Test

a = RNN([46,72,10],output = "Sigmoid")
a.train(TESTFILE,15,700)
