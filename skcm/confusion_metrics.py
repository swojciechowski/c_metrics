import numpy as np

class ConfusionMetrics:
    def __init__(self, confusion_matrix):
        if confusion_matrix.shape != (2, 2):
            raise ValueError("Invalid confusion matrix. Only binary problems are supported.")
        
        self.cm = confusion_matrix

    @property
    def TP(self):
        # True Positive
        return self.cm[0, 0]

    @property
    def FN(self):
        # False Negative
        return self.cm[0, 1]
    
    @property
    def FP(self):
        # False Positive
        return self.cm[1, 0]
    
    @property
    def TN(self):
        # True Negative
        return self.cm[1, 1]
    
    @property
    def AP(self):
        # Actual Positive
        return self.TP + self.FN
    
    @property
    def AN(self):
        # Actual Negative
        return self.FP + self.TN
    
    @property
    def PP(self):
        # Predicted Positive
        return self.TP + self.FP
    
    @property
    def PN(self):
        # Predicted Negative
        return self.FN + self.TN

    @property
    def TPR(self):
        # True Positive Rate
        return self.TP / self.P

    @property
    def FNR(self):
        # False Negative Rate
        return 1 - self.TPR

    @property
    def TNR(self):
        # True Negative Rate
        return self.TN / self.N

    @property
    def FPR(self):
        # False Positive Rate
        return 1 - self.TNR

    @property
    def PPV(self):
        # Positive Predictive Value
        return self.TP / self.PP
    
    @property
    def FDR(self):
        # False Discovery Rate
        return 1 - self.PPV

    @property
    def NPV(self):
        # Negative Predictive Value
        return self.TN / self.PN

    @property
    def FOR(self):
        # False Ommision Rate
        return 1 - self.NPV
    
    @property
    def PLR(self):
        # Positive Likehood Ratio
        return self.TPR / self.FPR

    @property
    def NLR(self):
        # Negative Likehood Ratio
        return self.FNR / self.TNR
    
    @property
    def DOR(self):
        # Diagnostic Odds Ratio
        return self.PLR / self.NLR

    @property
    def total_population(self):
        return self.AP + self.AN
    
    @property
    def prevalence(self):
        return self.P / (self.P + self.N)

    @property
    def markednes(self):
        return self.PPV + self.NPV - 1

    @property
    def accuracy(self):
        return (self.TP + self.TN) / (self.P + self.N)

    @property
    def balanced_accuracy(self):
        return (self.TPR + self.TNR) / 2

    @property
    def f1_score(self):
        return (2 * self.PPV * self.TPR) / (self.PPV + self.TPR)

    @property
    def gmean(self):
        return np.sqrt(self.TPR * self.TNR)
    
    @property
    def fowlkes_mallows_index(self):
        return np.sqrt(self.PPV * self.TPR)
    
    @property
    def matthews_correlation_coefficient(self):
        return np.sqrt(
            (self.TPR * self.TNR * self.PPV * self.NPV) - 
            (self.FNR * self.FPR * self.FOR * self.FDR)
        )

    @property
    def jaccard_index(self):
        return self.TP / (self.TP + self.FN + self.FP)
