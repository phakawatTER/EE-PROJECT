import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
import os 
import datetime
from find_day import find_weekday_order_number
plt.rcParams["figure.figsize"] = (20,6)
current_dir = os.path.dirname(__file__) 

class plot_load:
    
    def __init__(self,date=datetime.date.today(),fill=True,scatter=False,daydelta=0,update_interval=20):
        timedelta = datetime.timedelta(days=daydelta)
        self.date = date +timedelta
        self.fill = fill
        self.scatter = scatter
        self.data=[]
        fig = plt.figure()
        self.ax1 = fig.add_subplot(1,2,1)
        self.ax2 = fig.add_subplot(1,2,2)
        # self.ax2.plot(range(200),range(200),color="blue")
        # self.ax2.plot(range(200),range(400,600),color="green")
        ani = animation.FuncAnimation(fig,self.animate,interval=update_interval)
        plt.show()
        
    
        
    def animate(self,i):
        def get_weekday(day_index):
            weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            return weekdays[day_index]
        try:
            graph_data = pd.read_csv(os.path.join(current_dir,"graph_data","load_{}.csv".format(self.date)))
            self.ax1.clear()
            # self.ax2.clear()
            self.ax1.grid(True)
            self.ax2.grid(True)
            day_index = self.date.weekday()
            weekday = get_weekday(day_index)
            weekday_index_in_month =  find_weekday_order_number(self.date)+1
            self.ax1.title.set_text("Load Prediction for \"{}({}) {}\"".format(weekday,weekday_index_in_month,self.date))
            self.ax2.title.set_text("Cost Estimation")
            plt.setp(self.ax1.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
            self.ax1.plot(graph_data["time"],graph_data["load"])
            self.ax1.set_xlabel('datetime')
            self.ax1.set_ylabel('load')
            self.ax2.set_xlabel("datatime")
            self.ax2.set_ylabel("cost")
            if self.fill: # fill area under the curve
                min_load = min(graph_data["load"].values)
                self.ax1.fill_between(graph_data["time"],graph_data["load"],min_load,alpha=1,color="orange")
            if self.scatter: # plot scatter point
                self.ax1.scatter(graph_data["time"],graph_data["load"],color="b")
            if len(graph_data["time"].values) > 20: 
                for i,label in enumerate(self.ax1.get_xaxis().get_ticklabels()):
                    if i%int(len(graph_data["time"].values)*0.20)!=0:
                        label.set_visible(False)
        except Exception as err:
            print(err)

if __name__ == "__main__":
    import argparse
    def str2bool(string):
        string = string.lower()
        if string in ["true","1","t"]:
            return True
        else:
            return False
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",required=False,default=datetime.date.today(),help="load prediction date")
    ap.add_argument("--daydelta",required=False,default=0,type=int,help="daydelta")
    ap.add_argument("--scatter-plot",required=False,default=True,type=str2bool,help="scatter")
    ap.add_argument("--fill-plot",required=False,default=True,type=str2bool,help="fill plot")
    args = vars(ap.parse_args())
    date = args["date"]
    if date:
        date = datetime.datetime.strptime(date,"%Y-%m-%d").date()
    print(date)
    daydelta = args["daydelta"]
    plot_load(date=date,daydelta=daydelta)