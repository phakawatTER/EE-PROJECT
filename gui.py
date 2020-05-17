import tkinter as tk
from tkinter import messagebox
from predict import predictor
from tkcalendar import Calendar, DateEntry
from datetime import datetime
from plot_load import plot_load
import time
import os
import sys
import subprocess
import pickle

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
        self.ignore_warning = False
        self.model = os.path.join(current_dir,"model/m.h5")
        self.graph_process = None
        self.predictor_process = None 
        tk.Tk.__init__(self)
        self.resizable(False,False)
        self.title("Load Predictor")
        num_biomass = 3
        num_biogas = 2
        num_solar = 2
        biomass_pv = 12.35
        biogas_pv = 14
        # Try to load previous configuration
        try:
            config_path = os.path.join(current_dir,"config.pickle")
            with open(config_path,"rb") as r:
                config = pickle.load(r)
                num_biomass = config["num_biomass"]
                num_biogas = config["num_biogas"]
                num_solar = config["num_solar"]
                biomass_pv = config["biomass_pv"]
                biogas_pv = config["biogas_pv"]
            
        except Exception as err:
            print(err)
        
        
        self.options = {
            "num_biomass":tk.IntVar(self,value=num_biomass),
            "num_biogas":tk.IntVar(self,value=num_biogas),
            "num_solar":tk.IntVar(self,value=num_solar),
            "biomass_pv":tk.DoubleVar(self,value=biomass_pv),
            "biogas_pv":tk.DoubleVar(self,value=biogas_pv),
            "use_gpu":tk.IntVar(self,value=0),
            "scatter":tk.IntVar(self,value=0),
            "fill":tk.IntVar(self,value=1)
        }
        self.mainframe = tk.Frame(self,bg=GUI.PRIMARY_COLOR)
        self.mainframe.config(width=300,height=250)
        self.mainframe.pack()
        self.add_option() # plot option frame
        self.add_calendar() # add calendar 
        self.add_generator_option() # add generator option frame
        self.add_control_button() # add start prediction button
        def on_close():
            try:
                subprocess.Popen.kill(self.graph_process)
            except:
                pass
            try:
                self.stop_prediction()
            except:
                pass
            self.destroy()     
        self.protocol("WM_DELETE_WINDOW", on_close)
        self.mainloop()
        
    def power_error_callback(self,title,message): # handle power failure event
        if not self.ignore_warning:
            """
                Yes -> True
                No -> False
                Cancel -> None
            """       
            value = messagebox.askokcancel(title,message)
            if value == True:
                self.pred.should_terminate = True # force to terminate 
                try:
                    subprocess.Popen.kill(self.graph_process)
                except:
                    pass
                
    def add_option(self):
        self.option_frame = tk.Frame(self.mainframe,bg=GUI.PRIMARY_COLOR)
        self.option_frame.grid(row=2,column =0,sticky="nwe")
        
        
        tk.Label(self.option_frame,text=" TF Option",bg=GUI.PRIMARY_COLOR).pack(anchor="nw")
        tk.Checkbutton(self.option_frame, 
                  text="Use GPU",
                  variable=self.options["use_gpu"],
                  bg=GUI.PRIMARY_COLOR).pack(anchor="nw")
        
        tk.Label(self.option_frame,text=" Plot Option",bg=GUI.PRIMARY_COLOR).pack(anchor="nw")
        tk.Checkbutton(self.option_frame, 
                  text="Scatter Plot",
                  variable=self.options["scatter"],
                #   command=lambda : toggle("scatter") ,
                  bg=GUI.PRIMARY_COLOR).pack(anchor="nw")

        tk.Checkbutton(self.option_frame, 
                  text="Fill Plot",
                  variable=self.options["fill"],
                #   command=lambda : toggle("scatter") ,
                  bg=GUI.PRIMARY_COLOR).pack(anchor="nw")
        tk.Label(self.option_frame,text=" Predict Interval (ms)",bg=GUI.PRIMARY_COLOR).pack(anchor="w")
        self.interval_scaler = tk.Scale(self.option_frame, from_=0, to=2500, orient=tk.HORIZONTAL,bg=GUI.PRIMARY_COLOR)
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
        
    def add_generator_option(self):
        self.gen_option_frame = tk.Frame(self.mainframe,bg=GUI.PRIMARY_COLOR)
        self.gen_option_frame.grid(row=3,column=1)
        # ADD SCALERS COLUMN 0
        tk.Label(self.gen_option_frame,text="Biomass Generator",bg=GUI.PRIMARY_COLOR).grid(row=1,column=0)
        tk.Scale(self.gen_option_frame, from_=0, to=15, orient=tk.HORIZONTAL,variable=self.options["num_biomass"],bg=GUI.PRIMARY_COLOR).grid(row=2,column=0)
       
        tk.Label(self.gen_option_frame,text="Biogas Generator",bg=GUI.PRIMARY_COLOR).grid(row=3,column=0)
        tk.Scale(self.gen_option_frame, from_=0, to=15, orient=tk.HORIZONTAL,variable=self.options["num_biogas"],bg=GUI.PRIMARY_COLOR).grid(row=4,column=0)
        
        tk.Label(self.gen_option_frame,text="Solar Cell Generator",bg=GUI.PRIMARY_COLOR).grid(row=5,column=0)
        tk.Scale(self.gen_option_frame, from_=0, to=15, orient=tk.HORIZONTAL,variable=self.options["num_solar"],bg=GUI.PRIMARY_COLOR).grid(row=6,column=0)
        
        
        # ADD SCALERS COLUMN 1
        tk.Label(self.gen_option_frame,text="Biomass Gen. Power (kWH)",bg=GUI.PRIMARY_COLOR).grid(row=1,column=1)
        tk.Scale(self.gen_option_frame, from_=0, to=100,digits=4,resolution =   0.01, orient=tk.HORIZONTAL,variable=self.options["biomass_pv"],bg=GUI.PRIMARY_COLOR).grid(row=2,column=1)
       
        tk.Label(self.gen_option_frame,text="Biogas Gen. Power (kWH)",bg=GUI.PRIMARY_COLOR).grid(row=3,column=1)
        tk.Scale(self.gen_option_frame, from_=0, to=100,digits=4,resolution =   0.01, orient=tk.HORIZONTAL,variable=self.options["biogas_pv"],bg=GUI.PRIMARY_COLOR).grid(row=4,column=1)
        
        
    def stop_prediction(self):
        try:
            self.pred.should_terminate = True
        except:
            pass
        try:
            subprocess.Popen.kill(self.graph_process)
        except:
            pass
        
        
    def save_config(self):
        config = {
            "num_biomass": self.options["num_biomass"].get(),
            "num_biogas":self.options["num_biogas"].get(),
            "num_solar":self.options["num_solar"].get(),
            "biomass_pv":self.options["biomass_pv"].get(),
            "biogas_pv":self.options["biogas_pv"].get()
        }
        with open('config.pickle', 'wb') as f:
            pickle.dump(config, f)
        messagebox.showinfo("Save","Save Configuration Succeeded!")
        

    def add_control_button(self):
        self.control_button_frame = tk.Frame(self.mainframe,bg=GUI.PRIMARY_COLOR)
        self.control_button_frame.grid(row=2,column=2,sticky="NS")
        
        self.start_button = tk.Button(self.control_button_frame,text="Start Predicting")
        self.start_button.config(command=self.predict)
        self.start_button.grid(row=0,column=0,sticky='nesw')
        
        self.stop_button = tk.Button(self.control_button_frame,text="Stop Predicting")
        self.stop_button.config(command=self.stop_prediction)
        self.stop_button.grid(row=1,column=0,sticky='nesw')
        
        self.save_config_button = tk.Button(self.control_button_frame,text="Save\nConfiguration")
        self.save_config_button.config(command=self.save_config)
        self.save_config_button.grid(row=2,column=0,sticky='nesw')
        
        
        self.control_button_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        # self.control_button_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        self.control_button_frame.grid_rowconfigure(0, weight=1)
        self.control_button_frame.grid_rowconfigure(1, weight=1)
        self.control_button_frame.grid_rowconfigure(2, weight=1)
        
    def predict(self):
        self.ignore_warning = False
        try:
            self.pred.should_terminate = True
            print("Successfully terminated predicting thread !")
        except Exception as err:
            print(err)
        
        # KILL EXISTING PROCESS
        try:
            subprocess.Popen.kill(self.graph_process)
        except Exception as err:
            print(err)
        
        date = (self.calendar.get_date())
        date = datetime.strptime(date,"%Y-%m-%d")
        import threading as th
        import multiprocessing as mp
        daydelta = 0
        fill_plot=bool(self.options["fill"].get()) # fill plot
        scatter_plot = bool(self.options["scatter"].get()) # use scatter plot
        print("plot options",fill_plot,scatter_plot)
        use_gpu = bool(self.options["use_gpu"].get()) # use gpu for tensorflow 
        num_biomass = self.options["num_biomass"].get()
        num_biogas = self.options["num_biogas"].get()
        num_solar = self.options["num_solar"].get()
        biomass_pv = self.options["biomass_pv"].get()
        biogas_pv = self.options["biogas_pv"].get()
        period = 15
        self.pred = predictor(dt=date,model_path=self.model,use_gpu=use_gpu,message_callback=self.power_error_callback) # creat new instance of predictor
        self.pred.iteration_delay = self.interval_scaler.get()/1000 # self.predictor loop delay
        self.pred.BIOMASS_PV = biomass_pv
        self.pred.BIOGAS_PV = biogas_pv
        self.pred.num_biomass = num_biomass
        self.pred.num_biogas = num_biogas
        self.pred.num_solar = num_solar
        pred_thread = th.Thread(target=self.pred.run)
        pred_thread.start()
        self.graph_process = subprocess.Popen([
            "python",
            "plot_load.py",
            "--date",str(date.date()),
            "--scatter-plot", str(scatter_plot),
            "--fill-plot", str(fill_plot)
        ])
        
if __name__ == "__main__":
    GUI()