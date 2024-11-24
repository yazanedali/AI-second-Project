import random
from acivarionFunctions import *
import matplotlib.pyplot as plt

def SSE(prediction,actual):
    sum=0
    for i in range(len(actual)):
        sum+=(prediction[i][0]-actual[i])**2
    return sum
def MSE(prediction,actual):
    if(len(actual)==0):
        return
    return SSE(prediction,actual)/len(actual)
class Perceptron:
  def __init__(self, learning_rate=0.1, epochs=100):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weights = None
    self.bias = None
  def shortDistance(self,coor):
        return (sum(w * x for w, x in zip(self.weights, coor)) + self.bias)/(self.weights[0]**2 +self.weights[1]**2)**0.5


  def fit(self, X, y,val_set,activation_function=step):
    self.activation_function=activation_function


    n_samples = len(X)
    n_features = len(X[0])
    self.weights = [random.uniform(-0.5,0.5) for _ in range(n_features)]
    self.bias = 0
    patience = 5
    best_val_loss =1000000000
    for _ in range(self.epochs):
      errors = 0
      learning_rate = self.learning_rate

      for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]
        prediction = self.predict([x_i])

        if prediction != y_i:
          errors += 1
          update = learning_rate * (y_i - prediction[0])
          for j in range(n_features):
            self.weights[j] += update * x_i[j]
          self.bias += update

          learning_rate *= 1 + (errors / n_samples)
      predictions_val =[self.predict( [val_set[j][:2]]) for j in range(len(val_set)) ]
      
      val_loss = MSE(predictions_val, [val_set[j][2] for j in range(len(val_set)) ])
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 50
      else:
        patience -= 1
        if patience == 0:
            break
        
    

  def predict(self, X):

    res=[]
    for i in X:
      activation = sum(w * x for w, x in zip(self.weights, i)) + self.bias
      res.append(self.activation_function(activation))
    return res
  
  def evaluate(self,data):
       predictions = self.predict([data[i][:2] for i in range(len(data))])
       c=0
       for i in range(len(predictions)):
            if predictions[i]==data[i][2]:
                c+=1
       return(c*100/len(data))


class MultiClassPerceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.perceptrons = None

    def fit(self, X, y,val_set,activation_function=step):
        n_samples = len(X)
        n_features = len(X[0])
        classes = list(set(y))
        self.perceptrons = [Perceptron(self.learning_rate, self.epochs) for _ in classes]

  
        for j, perceptron in enumerate(self.perceptrons):
              target = [0] * len(y)
              target = [1 if classes[j] == y[i] else 0 for i in range(len(y))]
              val= [[val_set[i][0],val_set[i][1],1] if classes[j] == val_set[i][2] else[val_set[i][0],val_set[i][1],0] for i in range(len(val_set))]
             
              perceptron.fit(X, target,val)
   
    def predict(self, inputs):
        ans=[]

        for X in inputs:

            activations = [perceptron.predict([X])[0] for perceptron in self.perceptrons]
            ans.append( activations.index(max(activations)))
            if max(activations)==0:
                activations=[perceptron.shortDistance(X) for perceptron in self.perceptrons]
                ans[-1]=activations.index(min(activations))
        return ans
    

    def evaluate(self,data):
       predictions = self.predict([data[i][:2] for i in range(len(data))])
       c=0
       for i in range(len(predictions)):
            if predictions[i]==data[i][2]:
                c+=1
       return(c*100/len(data))


def get_decision_boundary(weights, bias,xmin,xmax):


  if weights[0] == 0:
    return None 

  slope = -weights[1] / weights[0]

  x1_range = range(xmin, xmax)
  boundary_points = [(x, slope * x - bias / weights[0]) for x in x1_range]
  return boundary_points
def split(data,train_ratio=0.7,test_ratio=0.15,val_ratio=0.15):
    
    total_length=len(data)
    first_marger=int(total_length*train_ratio)
    second_margier=int(total_length*(train_ratio+val_ratio))
    
    return data[:first_marger],data[first_marger:second_margier],data[second_margier:]

def split_randomly(data,train_ratio=0.7,test_ratio=0.15,val_ratio=0.15):
    total_length=len(data)
    first_marger=int(total_length*train_ratio)
    second_margier=int(total_length*(train_ratio+val_ratio))

    random.shuffle(data)
    set1 = data[:first_marger]
    set2 = data[first_marger:second_margier]
    set3 = data[second_margier:]
    
    
    return set1,set2,set3

def main():
  data=[
   [0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[0,1,0],[1,1,0],[2,1,0],[3,1,0],[1,2,0],[2,2,0],[3,2,0],[4,2,0],[0,3,0],[1,3,0],[2,3,0],[3,3,0],[4,3,0],[0,4,0],[1,4,0],[2,4,0],[3,4,0],[4,4,0],
   [0.5,0,0],[1.5,0,0],[2.5,0,0],[3.5,0,0],[4.5,0,0],[0.5,1,0],[1.5,1,0],[2.5,1,0],[0.5,2,0],[1.5,2,0],[2.5,2,0],[3.5,2,0],[4.5,2,0],[0.5,3,0],[1.5,3,0],[2.5,3,0],[3.5,3,0],[4.5,3,0],[0.5,4,0],[1.5,4,0],[2.5,4,0],[3.5,4,0],[4.5,4,0],
   [0,0.5,0],[1,0.5,0],[2,0.5,0],[3,0.5,0],[4,0.5,0],[0,1.5,0],[1,1.5,0],[2,1.5,0],[0,2.5,0],[1,2.5,0],[2,2.5,0],[3,2.5,0],[4,2.5,0],[0,3.5,0],[1,3.5,0],[2,3.5,0],[3,3.5,0],[4,3.5,0],[0,4.5,0],[1,4.5,0],[2,4.5,0],[3,4.5,0],[4,4.5,0],
   [0.5,0.5,0],[1.5,0.5,0],[2.5,0.5,0],[3.5,0.5,0],[4.5,0.5,0],[0.5,1.5,0],[1.5,1.5,0],[4.5,1.5,0],[0.5,2.5,0],[1.5,2.5,0],[2.5,2.5,0],[3.5,2.5,0],[4.5,2.5,0],[0.5,3.5,0],[1.5,3.5,0],[2.5,3.5,0],[3.5,3.5,0],[4.5,3.5,0],[0.5,4.5,0],[1.5,4.5,0],[2.5,4.5,0],[3.5,4.5,0],[4.5,4.5,0],

   [10,10,1],[11,10,1],[12,10,1],[13,10,1],[14,10,1],[10,11,1],[11,11,1],[14,11,1],[10,12,1],[11,12,1],[12,12,1],[13,12,1],[14,12,1],[10,13,1],[11,13,1],[12,13,1],[13,13,1],[14,13,1],[10,14,1],[11,14,1],[12,14,1],[13,14,1],[14,14,1],
   [10.5,10,1],[11.5,10,1],[12.5,10,1],[13.5,10,1],[14.5,10,1],[10.5,11,1],[13.5,11,1],[14.5,11,1],[10.5,12,1],[11.5,12,1],[12.5,12,1],[13.5,12,1],[14.5,12,1],[10.5,13,1],[11.5,13,1],[12.5,13,1],[13.5,13,1],[14.5,13,1],[10.5,14,1],[11.5,14,1],[12.5,14,1],[13.5,14,1],[14.5,14,1],
   [10,10.5,1],[11,10.5,1],[12,10.5,1],[13,10.5,1],[14,10.5,1],[10,11.5,1],[13,11.5,1],[14,11.5,1],[10,12.5,1],[11,12.5,1],[12,12.5,1],[13,12.5,1],[14,12.5,1],[10,13.5,1],[11,13.5,1],[12,13.5,1],[13,13.5,1],[14,13.5,1],[10,14.5,1],[11,14.5,1],[12,14.5,1],[13,14.5,1],[14,14.5,1],
   [10.5,10.5,1],[11.5,10.5,1],[12.5,10.5,1],[13.5,10.5,1],[14.5,10.5,0],[12.5,11.5,1],[13.5,11.5,1],[14.5,11.5,1],[10.5,12.5,1],[11.5,12.5,1],[12.5,12.5,1],[13.5,12.5,1],[14.5,12.5,1],[10.5,13.5,1],[11.5,13.5,1],[12.5,13.5,1],[13.5,13.5,1],[14.5,13.5,1],[10.5,14.5,1],[11.5,14.5,1],[12.5,14.5,1],[13.5,14.5,1],[14.5,14.5,1],
   
   [10,0,2],[11,0,2],[12,0,2],[13,0,2],[14,0,2],[10,1,2],[11,1,2],
   [12,1,2],[13,1,2],[11,2,2],[12,2,2],[13,2,2],[14,2,2],[10,3,2],
   [11,3,2],[12,3,2],[13,3,2],[14,3,2],[10,4,2],[11,4,2],[12,4,2],[13,4,2],[14,4,2],
   [10.5,0,2],[11.5,0,2],[12.5,0,2],[13.5,0,2],[14.5,0,2],[10.5,1,2],
   [11.5,1,2],[12.5,1,2],[13.5,1,2],[14.5,1,2],[10.5,2,2],[11.5,2,2],
   [14.5,2,2],[10.5,3,2],[11.5,3,2],[12.5,3,2],[13.5,3,2],[14.5,3,2],
   [10.5,4,2],[11.5,4,2],[12.5,4,2],[13.5,4,2],[14.5,4,2],
   [10,0.5,2],[11,0.5,2],[12,0.5,2],[13,0.5,2],[14,0.5,2],[10,1.5,2],
   [11,1.5,2],[12,1.5,2],[10,2.5,2],[11,2.5,2],[12,2.5,2],[13,2.5,2],
   [14,2.5,2],[10,3.5,2],[11,3.5,2],[12,3.5,2],[13,3.5,2],[14,3.5,2],
   [10,4.5,2],[11,4.5,2],[12,4.5,2],[13,4.5,2],[14,4.5,2],
   [10.5,0.5,2],[11.5,0.5,2],[12.5,0.5,2],[13.5,0.5,2],[14.5,0.5,2],
   [10.5,1.5,2],[11.5,1.5,2],[12.5,1.5,2],[10.5,2.5,2],[11.5,2.5,2],[12.5,2.5,2],[13.5,2.5,2],[14.5,2.5,2],[10.5,3.5,2],[11.5,3.5,2],[12.5,3.5,2],[13.5,3.5,2],[14.5,3.5,2],[10.5,4.5,2],[11.5,4.5,2],[12.5,4.5,2],[13.5,4.5,2],[14.5,4.5,2],

   [0,10,3],[1,10,3],[2,10,3],[3,10,3],[4,10,3],[0,11,3],[1,11,3],[2,11,3],[3,11,3],
   [4,11,3],[0,12,3],[1,12,3],[2,12,3],[3,12,3],[4,12,3],[0,13,3],[1,13,3],[2,13,3],[3,13,3],[4,13,3],[0,14,3],[1,14,3],[2,14,3],                                                                                             
   [0.5,10,3],[1.5,10,3],[2.5,10,3],[3.5,10,3],[4.5,10,3],[0.5,11,3],[1.5,11,3],[2.5,11,3],[3.5,11,3],[4.5,11,3],[0.5,12,3],[1.5,12,3],[2.5,12,3],[3.5,12,3],[4.5,12,3],[0.5,13,3],[1.5,13,3],[2.5,13,3],[3.5,13,3],[4.5,13,3],[0.5,14,3],[1.5,14,3],[2.5,14,3],                                              
   [0,10.5,3],[1,10.5,3],[2,10.5,3],[3,10.5,3],[4,10.5,3],[0,11.5,3],[1,11.5,3],[2,11.5,3],[3,11.5,3],[4,11.5,3],[0,12.5,3],[1,12.5,3],[2,12.5,3],[3,12.5,3],[4,12.5,3],[0,13.5,3],[1,13.5,3],[2,13.5,3],[3,13.5,3],[4,13.5,3],[0,14.5,3],[1,14.5,3],[2,14.5,3],                                              
   [0.5,10.5,3],[1.5,10.5,3],[2.5,10.5,3],[3.5,10.5,3],[4.5,10.5,3],[0.5,11.5,3],[1.5,11.5,3],[2.5,11.5,3],[3.5,11.5,3],[4.5,11.5,3],[0.5,12.5,3],[1.5,12.5,3],[2.5,12.5,3],[3.5,12.5,3],[4.5,12.5,3],[0.5,13.5,3],[1.5,13.5,3],[2.5,13.5,3],[3.5,13.5,3],[4.5,13.5,3],[0.5,14.5,3],[1.5,14.5,3],[2.5,14.5,3]
  ]

  X = [[data[i][0],data[i][1]] for i in range(len(data))]
  y = [ data[j][2]  for j in range(len(data))]

  multi_perceptron =MultiClassPerceptron(learning_rate=0.01, epochs=10000)
  multi_perceptron.fit(X, y,val_set=[[4,1,0],[0,2,0],
     [3.5,1,0],[4.5,1,0],
     [3,1.5,0],[4,1.5,0],
     [2.5,1.5,0],[3.5,1.5,0],

    [12,11,1],[13,11,1],
    [11.5,11,1],[12.5,11,1],
    [11,11.5,1],[12,11.5,1],
    [10.5,11.5,1],[11.5,11.5,1]
       ,[14,1,2],[10,2,2]
    ,[12.5,2,2],[13.5,2,2],
    [13,1.5,2],[14,1.5,2],
    [13.5,1.5,2],[14.5,1.5,2],

    [3.5,14.5,3],[4.5,14.5,3]
    ,[3,14,3],[4,14,3],       
    [3.5,14,3],[4.5,14,3],   
    [3,14.5,3],[4,14.5,3],  ]
    )

  test=[
     [4,1,0],[0,2,0],
     [3.5,1,0],[4.5,1,0],
     [3,1.5,0],[4,1.5,0],
     [2.5,1.5,0],[3.5,1.5,0],

    [12,11,1],[13,11,1],
    [11.5,11,1],[12.5,11,1],
    [11,11.5,1],[12,11.5,1],
    [10.5,11.5,1],[11.5,11.5,1]
       ,[14,1,2],[10,2,2]
    ,[12.5,2,2],[13.5,2,2],
    [13,1.5,2],[14,1.5,2],
    [13.5,1.5,2],[14.5,1.5,2],

    [3.5,14.5,3],[4.5,14.5,3]
    ,[3,14,3],[4,14,3],       
    [3.5,14,3],[4.5,14,3],   
    [3,14.5,3],[4,14.5,3],  
  ]

  plt.figure()

  for i in data:
     if i[2]==0:
          plt.scatter(i[0],i[1],color='red')
     if i[2]==1:
          plt.scatter(i[0],i[1],color='blue')
     if i[2]==3:
          plt.scatter(i[0],i[1],color='green')
     if i[2]==2:
          plt.scatter(i[0],i[1],color='yellow')


  colors=['red','blue','yellow','green']
  for i in range(4):
    boundary_points = get_decision_boundary(multi_perceptron.perceptrons[i].weights,multi_perceptron.perceptrons[i].bias,0,10)

    if boundary_points:
        x_boundary, y_boundary = zip(*boundary_points)
        plt.plot(x_boundary, y_boundary, color=colors[i],label="Decision Boundary")


  plt.show()

  print(multi_perceptron.evaluate(test))




if __name__=="__main__":
   main()