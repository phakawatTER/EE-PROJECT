import tkinter as tk
from predict import predictor
from tkcalendar import Calendar, DateEntry
from datetime import datetime
from plot_load import plot_load
import time
import os
import sys
import subprocess

current_dir = os.path.dirname(__file__)

class GUI(tk.Tk):
    PRIMARY_COLOR = "#fff"
    SECONDARY_COLOR = "#BDBDBD"
    BUTTON_COLOR = "#FF5733"
    NOW = datetime.now()
    DAY= NOW.day
    MONTH = NOW.month
    YEAR = NOW.year
    # PREDICTOR = predictor()
    def __init__(self):
        self.model = os.path.join(current_dir,"model/m.h5")
        self.graph_process = None
        self.predictor_process = None 
        tk.Tk.__init__(self)
        self.resizable(False,False)
        self.title("Load Predictor")
        self.plot_option = {
            "use_gpu":tk.IntVar(self,value=0),
            "scatter":tk.IntVar(self,value=0),
            "fill":tk.IntVar(self,value=1)
        }
        self.mainframe = tk.Frame(self,bg=GUI.PRIMARY_COLOR)
        self.mainframe.config(width=300,height=250)
        self.mainframe.pack()
        self.add_plot_option() # plot option frame
        self.add_calendar() # add calendar 
        self.add_start_button() # add start prediction button
        def on_close():
            subprocess.Popen.kill(self.graph_process)
            subprocess.Popen.kill(self.predictor_process)   
            self.destroy()     
        self.protocol("WM_DELETE_WINDOW", on_close)
        self.mainloop()
        
        

    def add_plot_option(self):

        self.option_frame = tk.Frame(self.mainframe,bg=GUI.PRIMARY_COLOR)
        self.option_frame.grid(row=2,column =0,sticky="nwe")
        tk.Label(self.option_frame,text=" TF Option",bg=GUI.PRIMARY_COLOR).pack(anchor="w")
        tk.Checkbutton(self.option_frame, 
                  text="Use GPU",
                  variable=self.plot_option["use_gpu"],
                  bg=GUI.PRIMARY_COLOR).pack(anchor="w")
        
        tk.Label(self.option_frame,text=" Plot Option",bg=GUI.PRIMARY_COLOR).pack(anchor="w")
        tk.Checkbutton(self.option_frame, 
                  text="Scatter Plot",
                  variable=self.plot_option["scatter"],
                #   command=lambda : toggle("scatter") ,
                  bg=GUI.PRIMARY_COLOR).pack(anchor="w")

        tk.Checkbutton(self.option_frame, 
                  text="Fill Plot",
                  variable=self.plot_option["fill"],
                #   command=lambda : toggle("scatter") ,
                  bg=GUI.PRIMARY_COLOR).pack(anchor="w")
        tk.Label(self.option_frame,text=" Predict Interval (ms)",bg=GUI.PRIMARY_COLOR).pack(anchor="w")
        self.interval_scaler = tk.Scale(self.option_frame, from_=100, to=2500, orient=tk.HORIZONTAL,bg=GUI.PRIMARY_COLOR)
        self.interval_scaler.pack()
        self.interval_scaler.set(250) # set default value
        

    def add_calendar(self):
        tk.Label(self.mainframe,text="Select Prediction Date",bg=GUI.PRIMARY_COLOR).grid(row=1,column=1,columnspan=2,sticky="W")
        self.calendar_frame = tk.Frame(self.mainframe,bg=GUI.SECONDARY_COLOR)
        self.calendar_frame.grid(row=2,column=1)
        self.calendar = Calendar(self.calendar_frame,
                   font="Arial 10", selectmode='day',
                   date_pattern="y-mm-dd",
                    year=GUI.YEAR, month=GUI.MONTH, day=GUI.DAY)
        self.calendar.pack(fill="both", expand=True)

    def add_start_button(self):
        self.start_button = tk.Button(
            self.mainframe,
        text="Predict"
        # ,bg=GUI.BUTTON_COLOR
        )
        self.start_button.grid(row=2,column=2,sticky="NS")
        self.start_button.config(command=self.predict)

    def predict(self):
        try:
            subprocess.Popen.kill(self.graph_process)
            subprocess.Popen.kill(self.predictor_process)
        except Exception as err:
            print(err)
        
        date = (self.calendar.get_date())
        date = datetime.strptime(date,"%Y-%m-%d")
        import threading as th
        import multiprocessing as mp
        daydelta = 0
        fill_plot=bool(self.plot_option["fill"].get()) # fill plot
        scatter_plot = bool(self.plot_option["scatter"].get()) # use scatter plot
        use_gpu = bool(self.plot_option["use_gpu"].get()) # use gpu for tensorflow 
        period = 15
        # pred = predictor(dt=date,model_path=self.model,use_gpu=use_gpu)
        # pred.iteration_delay = self.interval_scaler.get()/1000 # predictor loop delay
        # pred_thread = th.Thread(target=pred.run)
        # pred_thread.start()
        
        
        self.predictor_process = subprocess.Popen([
            "python",
            "predict.py",
            "--model",self.model,
            "--date",str(date.date()),
            "--gpu",str(use_gpu),
            "--show-graph",str(False)
        ])
        
        self.graph_process = subprocess.Popen([
            "python",
            "plot_load.py",
            "--date",str(date.date()),
            "--scatter-plot", str(scatter_plot),
            "--fill-plot", str(fill_plot)
        ])
        
if __name__ == "__main__":
    GUI()