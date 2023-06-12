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


# def Cosine(question_vector, sentence_vector):
#         dot_product = np.dot(question_vector, sentence_vector.T)
#         denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
#         return dot_product/denominator

# def Euclidean(question_vector, sentence_vector):
#         # vec1 = question_vector.copy()
#         # vec2 = sentence_vector.copy()
#         vec1 = torch.detach(question_vector)
#         vec2 = torch.detach(sentence_vector)
#         if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
#         vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
#         return np.linalg.norm(vec1-vec2)

# def Theta(question_vector, sentence_vector):
#         return np.arccos(Cosine(question_vector, sentence_vector)) + np.radians(10)

# def Triangle(question_vector, sentence_vector):
#         theta = np.radians(Theta(question_vector, sentence_vector))
#         return ((np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector)) * np.sin(theta))/2

# def Magnitude_Difference(vec1, vec2):
#         return abs((np.linalg.norm(vec1) - np.linalg.norm(vec2)))

# def Sector(question_vector, sentence_vector):
#         ED = Euclidean(question_vector, sentence_vector)
#         MD = Magnitude_Difference(question_vector, sentence_vector)
#         theta = Theta(question_vector, sentence_vector)
#         return math.pi * (ED + MD)**2 * theta/360

# question_vector = torch.rand(3, 128)
# sentence_vector = torch.rand(3, 128)

# # similarity_value = Triangle(question_vector,sentence_vector) * Sector(question_vector,sentence_vector)
# similarity_value = TS_SS(question_vector,sentence_vector).forward()
# # similarity_value = torch.tensor(similarity_value)
# print(similarity_value.shape)
# print(similarity_value)
# diag = torch.diag(similarity_value)
# print(diag.shape)
# print(diag)
