# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def tau(self, correct_w, w, fvector, c):
        norm = 0.0
        for i in fvector:
            norm += fvector[i]**2.0
        val = (((w-correct_w) * fvector)+1.0)/float(2*norm)
        return min(c,val)


    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        tmp_weights = []
        for c in range(len(Cgrid)):
            tmp = self.weights.copy()
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                    score = []
                    max_v = 0
                    for k,v in tmp.iteritems():
                       predict = trainingData[i] * v
                       score.append([k,predict])
                    max_v = score[0][1]
                    label = 0
                    for l in score:
                        if l[1] > max_v:
                            max_v = l[1]
                            label = l[0]
                   
                    if label != trainingLabels[i]:
                        t2 = self.tau(tmp[trainingLabels[i]], tmp[label], trainingData[i] ,Cgrid[c])
                        x = trainingData[i].copy()
                        for itr in x:
                            x[itr] *= t2
                        tmp[trainingLabels[i]] +=  x
                        tmp[label] -=  x
            tmp_weights.append(tmp)
        
        cp = 0
        for c in range(len(Cgrid)):
            max = 0
            w = tmp_weights[c]
            label = []  
            for i in range(len(validationData)):
                score = []
                for k,v in w.iteritems():
                    predict = validationData[i] * v
                    score.append([predict,k])
                max_label = reduce(lambda x,y: x if x[0]>y[0] else y, score)[1]
                label.append(max_label)

            accuracy = 0
            for i in range(len(label)):
                if label[i] == validationLabels[i]:
                    accuracy += 1
            accuracy = accuracy/float(len(label))
            if accuracy > max:
                cp = c
                max = accuracy
            if abs(accuracy-max) < 1:
                if Cgrid[c] < Cgrid[cp]:
                    cp = c
                    max = accuracy
        self.weights = tmp_weights[cp]


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


