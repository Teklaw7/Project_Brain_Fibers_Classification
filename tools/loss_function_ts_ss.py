import numpy as np
import math
import torch

class TS_SS:
        
    def __init__(self, question_vector, sentence_vector):
        self.question = question_vector
        self.sentence = sentence_vector
            
    def Cosine(self):
        dot_product = np.dot(self.question, self.sentence.T)
        denominator = (np.linalg.norm(self.question) * np.linalg.norm(self.sentence))
        return dot_product/denominator
    
    def Euclidean(self):
        # vec1 = self.question.copy()
        # vec2 = self.sentence.copy()
        vec1 = torch.detach(self.question)
        vec2 = torch.detach(self.sentence)
        if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
        vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
        return np.linalg.norm(vec1-vec2)
    
    def Theta(self):
        return np.arccos(self.Cosine()) + np.radians(10)
    
    def Triangle(self):
        theta = np.radians(self.Theta())
        return ((np.linalg.norm(self.question) * np.linalg.norm(self.sentence)) * np.sin(theta))/2
    
    def Magnitude_Difference(self):
        return abs((np.linalg.norm(self.question) - np.linalg.norm(self.sentence)))
    
    def Sector(self):
        ED = self.Euclidean()
        MD = self.Magnitude_Difference()
        theta = self.Theta()
        return math.pi * (ED + MD)**2 * theta/360
    
    def TS_SS(self):
        tri = self.Triangle()
        sec = self.Sector()
        return tri * sec
    
    def forward(self):
        return torch.tensor(self.TS_SS()), torch.diag(torch.tensor(self.TS_SS()))