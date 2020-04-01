from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
import os 

current_dir = os.path.dirname(__file__) 

class plot_load:
    
    def __init__(self):
        self.data=[]
        fig = plt.figure()
        self.ax1 = fig.add_subplot(1,1,1)
        ani = animation.FuncAnimation(fig,self.animate,interval=1000)
        plt.show()
    
    def animate(self,i):
        try:
            graph_data = pd.read_csv(os.path.join(current_dir,"graph_data","load.csv"))
            self.ax1.clear()
            self.ax1.tick_params(axis='x', labelrotation=90)
            self.ax1.plot(graph_data["time"],graph_data["load"])
            self.ax1.set_xlabel('datetime')
            self.ax1.set_ylabel('load')
            self.ax1.scatter(graph_data["time"],graph_data["load"],color="r")
        except:
            pass

if __name__ == "__main__":
    plot_load()