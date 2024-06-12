import numpy as np
from sklearn.metrics import f1_score

class ThreshMax:
    def __init__(self, classes=[0,1,2], thresh_class=1, class_to_optimize='avg'):
        self.classes = classes
        self.thresh_class = thresh_class
        self.class_to_optimize = class_to_optimize
        self.tmax = None
        self.fmax = None

    
    def compute_threshmax(self, y_pred, thresh):
        if thresh == None:
            thresh=self.tmax

        if y_pred[self.thresh_class] > thresh:
            return self.thresh_class
        else:
            y_pred[self.thresh_class] = 0 
            return np.argmax(y_pred)

    def find_tmax(self, y_preds, y_trues):
        y_decision_argmax = [np.argmax(y) for y in y_preds]
        if self.class_to_optimize=='avg':
            fmax = f1_score(y_trues, y_decision_argmax, average='macro')
        else:
            fmax = f1_score(y_trues, y_decision_argmax, average=None)[self.class_to_optimize]
        tmax = 0
        for t in np.unique(np.array(y_preds)[:,self.thresh_class]):
            y_decision = [self.compute_threshmax(y,thresh=t) for y in y_preds]
            if self.class_to_optimize == 'avg':
                f = f1_score(y_trues, y_decision, average='macro')
            else:
                f = f1_score(y_trues, y_decision, average=None)[self.class_to_optimize]
            if f > fmax:
                fmax = f
                tmax = t
        
        self.fmax = fmax
        self.tmax = tmax
        return tmax