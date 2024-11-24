import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from acivarionFunctions import *
from Models import *
from DrawRegions import *




class ClassificationApp:
    def __init__(self, master):

        self.abs_min_x=math.inf
        self.abs_max_x=-math.inf
        self.abs_min_y=math.inf
        self.abs_max_y=-math.inf

        self.colors=['#020f12','#05d7ff','#65e7ff','BLACK']
        self.master = master
        master.title("Classification App")
        master.configure(bg="#2c3e50")
       

        # Variables for user inputs
        self.data_points = []
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.max_iterations = tk.IntVar(value=1000)
        self.activation_function = tk.StringVar(value="sigmoid")
        self.class_names = ['Class 1', 'Class 2']
        self.class_colors=[]
        self.class_colors=[self.generate_random_color() for _ in self.class_names]
        self.num_classes = len(self.class_names)



        self.class_label = tk.Label(master, text="Select Class:")
        self.class_label.grid(row=1, column=0)
        self.class_label.configure(bg="#2c3e50",fg=self.colors[2])


        self.Xlabel=tk.Label(master,text= "X range")
        self.Xlabel.grid(row=2,column=2,columnspan=2)
        self.Xlabel.configure(bg="#2c3e50",fg=self.colors[2])

        self.x_min=tk.DoubleVar(value=0.0)
        self.X_min_box=tk.Entry(master,textvariable=self.x_min)
        self.X_min_box.grid(row=3,column=2,columnspan=1)
        
        self.x_max=tk.DoubleVar(value=10.0)
        self.x_max_box=tk.Entry(master,textvariable=self.x_max)
        self.x_max_box.grid(row=3,column=3,columnspan=1)

        self.y_label=tk.Label(master,text= "Y range")
        self.y_label.grid(row=4,column=2,columnspan=2)
        self.y_label.configure(bg="#2c3e50",fg=self.colors[2])
        
        self.y_min=tk.DoubleVar(value=0.0)
        self.y_min_box=tk.Entry(master,textvariable=self.y_min)
        self.y_min_box.grid(row=5,column=2,columnspan=1)

        
        self.y_max=tk.DoubleVar(value=10.0)
        self.y_max_box=tk.Entry(master,textvariable=self.y_max)
        self.y_max_box.grid(row=5,column=3,columnspan=1)

        self.x_max_box.bind("<Return>", self.updateRange)
        self.X_min_box.bind("<Return>", self.updateRange)
        self.y_max_box.bind("<Return>", self.updateRange)
        self.y_min_box.bind("<Return>", self.updateRange)

        self.selected_class = tk.StringVar(value=self.class_names[0])
        self.class_combo = ttk.Combobox(master,  textvariable=self.selected_class, values=self.class_names )
        self.class_combo.grid(row=1, column=1)

        self.add_class_button = tk.Button(master, text="Add Class", command=self.add_class)
        self.add_class_button.grid(row=1, column=2, padx=5, pady=5)
        self.clear_class_button = tk.Button(master, text="clear", command=self.clear)
        self.clear_class_button.grid(row=1, column=3, padx=5, pady=5)
        self.undo_button = tk.Button(master, text="undo", command=self.undo)
        self.undo_button.grid(row=1, column=4, padx=5, pady=5)

        self.learning_rate_label = tk.Label(master, text="Learning Rate:")
        self.learning_rate_label.grid(row=2, column=0)
        self.learning_rate_label.configure(bg="#2c3e50",fg=self.colors[2])

        self.learning_rate_entry = tk.Entry(master, textvariable=self.learning_rate)
        self.learning_rate_entry.grid(row=2, column=1)

        self.max_iterations_label = tk.Label(master, text="Max Iterations:")
        self.max_iterations_label.grid(row=3, column=0)
        self.max_iterations_label.configure(bg="#2c3e50",fg=self.colors[2])

        self.max_iterations_entry = tk.Entry(master, textvariable=self.max_iterations)
        self.max_iterations_entry.grid(row=3, column=1)

      
        self.activation_functions = ['Step', 'Sigmoid', 'ReLU','tanh','sign']
        self.selected_activation_function = tk.StringVar(value=self.activation_functions[0])

        
        self.activation_function_label = tk.Label(master, text="Activation Function:")
        self.activation_function_label.grid(row=4, column=0)
        self.activation_function_label.configure(bg="#2c3e50",fg=self.colors[2])


        self.activation_function_combo = ttk.Combobox(master, textvariable=self.selected_activation_function,
                                                      values=self.activation_functions)
        self.activation_function_combo.grid(row=4, column=1)
        # self.activation_function_combo.configure(background="#2c3e50")


        self.train_button = tk.Button(master,foreground=self.colors[3],activebackground=self.colors[2],activeforeground=self.colors[3],
                                      highlightthickness=2,highlightbackground=self.colors[1],highlightcolor='WHITE',
                                      width=13,height=2,border=0,cursor='hand1'
                                       ,background=self.colors[1],text="Train",relief=tk.FLAT,  command=self.train)
        self.train_button.grid(row=5, column=1, columnspan=1, padx=5, pady=5)
    
        
        self.performance_label = tk.Label(master, text="Performance:")
        self.performance_label.grid(row=6, column=1)
        self.performance_label.configure(background="#2c3e50",fg=self.colors[2])

        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=7, column=0, columnspan=5)

        self.shall_display_confusion_matrix=tk.IntVar()
        self.display_conf_matrix=tk.Checkbutton(master,text='display conusion matrix \n after training',variable=self.shall_display_confusion_matrix,onvalue=1,offvalue=0)
        self.display_conf_matrix.grid(row=5,column=0,rowspan=2)
        self.display_conf_matrix.configure(bg="#2c3e50",fg=self.colors[2])

        # Matplotlib event handling
        self.inlineDiagram=plt
        self.fig, self.ax = self.inlineDiagram.subplots()
        self.ax.set_title("Classification Regions")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax.grid(True)
        plt.xlim((float(self.X_min_box.get()),float(self.x_max_box.get())))
        plt.ylim((float (self.y_min_box.get()),float(self.y_max_box.get())))

    def updateRange(self,event=None):
       
        plt.xlim((float(self.X_min_box.get()),float(self.x_max_box.get())))
        plt.ylim((float (self.y_min_box.get()),float(self.y_max_box.get())))
        
        self.plot_data()

    def clear(self):
        self.ax.cla()  # Clear the current axes
        plt.xlim((float(self.X_min_box.get()), float(self.x_max_box.get())))  # Reset limits
        plt.ylim((float(self.y_min_box.get()), float(self.y_max_box.get())))
        self.canvas.draw()  # Redraw the canvas
        self.data_points=[]

    def undo(self):
        self.ax.cla()  # Clear the current axes
        plt.xlim((float(self.X_min_box.get()), float(self.x_max_box.get())))  # Reset limits
        plt.ylim((float(self.y_min_box.get()), float(self.y_max_box.get())))
        self.canvas.draw()  # Redraw the canvas
        self.data_points.pop()
        self.plot_data()


        pass
    def generate_random_color(self):
    # Generate random color
        while True:
            random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            if random_color not in  self.class_colors:
                        return random_color

                
        
    def on_click(self, event):
        x, y = event.xdata, event.ydata
        self.abs_min_x=min(self.abs_min_x,x)
        self.abs_max_x=max(self.abs_max_x,x)
        self.abs_min_y=min(self.abs_min_y,y)
        self.abs_max_y=max(self.abs_max_y,y)
        if x is not None and y is not None:
            class_index = self.class_combo.current()
            self.data_points.append([x, y, class_index])
            self.ax.clear()
            self.ax.set_title("Classification Regions")
            self.ax.set_xlabel("X1")
            self.ax.set_ylabel("X2")

            self.plot_data()

    def add_class(self):
        new_class_name = f"Class {self.num_classes + 1}"
        self.class_names.append(new_class_name)
        self.num_classes += 1
        self.class_combo['values'] = self.class_names
        self.class_colors.append(self.generate_random_color())
        messagebox.showinfo("Info", f"Added new class: {new_class_name}")

    def plot_data(self):
        self.ax.clear()
        self.ax.set_title("Classification Regions")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")

        for class_index in range(self.num_classes):
            class_data = [point for point in self.data_points if point[2] == class_index]
            if class_data:
                x = [point[0] for point in class_data]
                y = [point[1] for point in class_data]
                self.ax.scatter(x, y, color=self.class_colors[class_index], label=self.class_names[class_index])
            plt.xlim(((float(self.X_min_box.get()),float(self.x_max_box.get())))) 
            plt.ylim((float (self.y_min_box.get()),float(self.y_max_box.get())))
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def compute_confusion_matrix(self,test,classes):
         map=[]
         for _ in range(len(classes)):
           map.append([0 for _ in range(len(classes))])
         for i in range( len(test)):
            predicut=self.multi_perceptron.predict([test[i][:2]])
            map[test[i][2]][predicut[0]]+=1

         return map
         
         
    def display_confusion_matrix(self,test,classes):
        if classes==None:
            classes=self.class_names
        map=self.compute_confusion_matrix(test,classes)
        matrix=plt
        matrix.figure(figsize=(8, 6))
        matrix.imshow(map, interpolation='nearest', cmap=matrix.cm.Blues)
        matrix.title('Confusion Matrix')
        matrix.colorbar()
        tick_marks = [i for i in range(len(classes))]
        matrix.xticks(tick_marks, classes, rotation=45)
        matrix.yticks(tick_marks, classes)

   
        for i in range(len(classes)):
            for j in range(len(classes)):
                text_color = "white" if (i==j and map[i][j] !=0 ) else "black"
                matrix.text(j, i, format(map[i][j], "d"), ha="center", va="center", color=text_color)


        matrix.tight_layout()
        matrix.ylabel('True label')
        matrix.xlabel('Predicted label')

        # Show plot
        plt.show()
      
    def train(self):

        actFun=self.activation_function_combo.get()
        if actFun=='Step':
            activation_function=step
        elif actFun=='Sigmoid':
            activation_function=Sigmoid
        elif actFun=='ReLU':
            activation_function=Rlu
        elif actFun=='tanh':
            activation_function=tanh
        elif actFun=='sign':
            activation_function=sign
        

        data,valdition,test=split_randomly(self.data_points)
        X = [[data[i][0],data[i][1]] for i in range(len(data))]
        y = [ data[j][2]  for j in range(len(data))]

        if(len(self.class_colors)>2):
            self.multi_perceptron =MultiClassPerceptron(learning_rate=float (self.learning_rate.get()), epochs=int (self.max_iterations.get()))
            self.multi_perceptron.fit(X, y,val_set= valdition,activation_function=activation_function)

            performance=self.multi_perceptron.evaluate(test)
        else:
            self.multi_perceptron=Perceptron(learning_rate=float (self.learning_rate.get()), epochs=int (self.max_iterations.get()))
            self.multi_perceptron.fit(X, y,val_set= valdition,activation_function=activation_function)
            performance=self.multi_perceptron.evaluate(test)


        self.performance_label['text']="performance"+str(performance)
        self.plot_classification_region()
        if(self.shall_display_confusion_matrix.get()==1):
             self.display_confusion_matrix(test,self.class_names)
        self.plot_classification_region()

    def meshgrid_custom(self,x, y):

        X = [[val for val in x] for _ in y]
        Y = [[row for row in [yval] * len(x)] for yval in y]

        return X,Y
    def plot_classification_region_Distubution(self):
        external=plt
        x = [x for x in range(math.ceil(self.abs_min_x-1),math.ceil(self.abs_max_x)+2,1)]
        y = [x for x in range(math.ceil(self.abs_min_y)-1,math.ceil(self.abs_max_y)+2,1)]
        X, Y = self.meshgrid_custom(x, y) 

        points=[[(X[i][j],Y[i][j]) for j in range(len(X)) ]for i in range(len(X)) ]
        levels=[ i-0.5 for i in range(len(self.class_names)+1)]
        z=[ self.multi_perceptron.predict(point) for point in points]
      



        cs = external.contourf(X, Y, z, 
                  levels=levels,
                  colors=self.class_colors
				) 

        cbar = external.colorbar(cs) 
        cbar.set_ticks(levels)  # Set colorbar tick positions

        plt.show()
        external.disconnect()
    def  plot_classification_region(self):
        data=self.data_points
        X = [[data[i][0],data[i][1]] for i in range(len(data))]
        y = [ data[j][2]  for j in range(len(data))]

        

        feature1_min, feature1_max = min(X, key=lambda x: x[0])[0], max(X, key=lambda x: x[0])[0]
        feature2_min, feature2_max = min(X, key=lambda x: x[1])[1], max(X, key=lambda x: x[1])[1]
        step = 1/10000
        plot_classification_regions(X, y, self.multi_perceptron, feature1_min, feature1_max, feature2_min, feature2_max, step)
        





def main():
     root = tk.Tk()
     frame=tk.Frame(root,bg='#020f12')

   
    
     app = ClassificationApp(root)
    
     root.mainloop()

main()

