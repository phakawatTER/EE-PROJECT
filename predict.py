from  tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import numpy as np
import datetime 
import pandas as pd
import time
import os
import math
import sys
from find_day import find_weekday_order_number
from plot_load import plot_load


# set keras backend
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

current_dir = os.path.dirname(__file__)

class predictor:
    # Generating Power
    BIOMASS_PV = 12.35 # kWH
    BIOGAS_PV = 14 # kWH
    SOLAR_PV = 6.25  # kWH
    # Operation Cost
    BIOMASS_COST =  55.502 # Baht / W
    BIOGAS_COST =  70.00 # Baht / W
    SOLAR_COST =  99 # Baht / W
    OFFSET_FACTOR = 1.1 

    def __init__(self,dt=datetime.datetime.today(),model_path="",predict_period=15,use_gpu=True,daydelta=0):
        if not os.path.exists(os.path.join(current_dir,"graph_data")):
            os.mkdir(os.path.join(current_dir,"graph_data"))
        self.should_terminate = False
        self.start_time = time.time() # start prediction time
        self.dataframe = pd.DataFrame(columns=["time","load"])
        self.predict_period = math.ceil(predict_period)
        self.iteration_delay = 0.1
        timedelta = datetime.timedelta(days=daydelta)
        dt = dt+timedelta # get today datetime
        self.date = dt.date()
        self.start_datetime = dt.replace(hour=0,minute=0,second=0,microsecond=0)
        self.start_day = self.start_datetime.day
        self.use_gpu = use_gpu

        self.current_load = 0
        self.optimal_data = None

        self.device = "/device:CPU:0"
        if self.use_gpu:
            self.device = "/GPU:0"
        with tf.device(self.device):
            self.model = load_model(model_path) # model for prediction


    def cal_optimal_cost(self,num_biomass=3,num_biogas=2,num_solar=2):
        time_portion = 4
        load = self.current_load * predictor.OFFSET_FACTOR # new load with offset
        operation_cost = 0
        all_cost = {}
        for i in range(time_portion+1):
            biomass_portion = i
            for j in range(time_portion+1):
                solar_portion = j
                for k in range(time_portion+1):
                    biogas_portion = k
                    # PRODUCTS
                    biomass_load =  num_biomass*(biomass_portion/time_portion)*self.predict_period/60*predictor.BIOMASS_PV # kW 
                    biogas_load =  num_biogas*(biogas_portion/time_portion)*self.predict_period/60*predictor.BIOGAS_PV # kW
                    solar_load = num_solar*(solar_portion/time_portion)*self.predict_period/60*predictor.SOLAR_PV # kW
                    # COST
                    biomass_cost = num_biomass * biomass_load * predictor.BIOMASS_COST * 1000
                    biogas_cost = num_biogas * biogas_load * predictor.BIOGAS_COST * 1000
                    solar_cost = num_solar * solar_load * predictor.SOLAR_COST * 1000

                    total_load = biomass_load + biogas_load + solar_load
                    total_cost = biomass_cost + biogas_cost + solar_cost 
                
                    # if total_load < load: # continue if generated load is 
                    #     continue
                    
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,biogas_portion,solar_portion)] = {}
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,biogas_portion,solar_portion)]["bm_cost"] = biomass_cost
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,biogas_portion,solar_portion)]["bg_cost"] = biogas_cost
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,biogas_portion,solar_portion)]["s_cost"] = solar_cost
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,biogas_portion,solar_portion)]["total_cost"] = total_cost
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,biogas_portion,solar_portion)]["load"] = total_load
                    
        # print("-"*200)
        # print("DATETIME:",self.start_datetime)
        # for key in all_cost:
        #     print(key,"--->",all_cost[key])
        # print("-"*200)

        return all_cost

    def get_predict_datetime(self,prediction_time):
        day = (prediction_time.weekday()+1)%7
        month = prediction_time.month
        hour = prediction_time.hour
        minute = prediction_time.minute
        return np.array([[day,month,hour,minute]]).astype(np.float32).reshape(-1,4)

    def predict_load(self,time):
        with tf.device(self.device):
            p_load = self.model.predict(time)
            return p_load

    def task(self):
        weekday_index_in_month = find_weekday_order_number(self.start_datetime)
        timedelta = datetime.timedelta(minutes=self.predict_period)
        self.start_datetime += timedelta
        current_day = self.start_datetime.day
        day = (self.start_datetime.weekday()+1)%7
        month = self.start_datetime.month
        hour = self.start_datetime.hour
        minute = self.start_datetime.minute
        input_vector = [day,weekday_index_in_month,month,hour,minute]
        tensor = np.array([input_vector]).astype(np.float32).reshape(-1,len(input_vector)) # input tensor
        load = self.predict_load(tensor) # predict load
        self.current_load = load[0][0] # current predicted load 
        self.cal_optimal_cost() # calculate optimal cost with the 2 generators
        # solar_cost =  optimal_cost["solar"]
        # biomass_cost = optimal_cost["biomass"]
        # cost = optimal_cost["cost"]
        # biomass_portion = optimal_cost["biomass_portion"] # optimal solar portion
        # solar_portion = optimal_cost["solar_portion"] # optimal solar portion
        df = pd.DataFrame(data={
            "time":[self.start_datetime],
            "load":[load[0][0]]
            # "cost":[cost],
            # "solar_cost":[solar_cost],
            # "biomass_cost":[biomass_cost],
            # "biomass_portion":[biomass_portion],
            # "solar_portion":[solar_portion]
            })
        print(df)
        self.dataframe=self.dataframe.append(df,ignore_index=True)
        self.dataframe.to_csv(os.path.join(current_dir,"graph_data","load_{}.csv".format(self.date)))
        
        if self.start_day != current_day:
            self.should_terminate = True
        print(f"{self.start_datetime} ----> {load[0][0]}")

    def run(self):
        while True:
            if not self.should_terminate:
                self.task()
                time.sleep(self.iteration_delay)
            else:
                break
        print(f"End Prediction on {self.date}")

if __name__=="__main__":
    # function to parse string to boolean
    def str2bool(v):
        v = v.lower()
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    import sys
    import argparse 
    import multiprocessing as mp
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model",required=False,help="Path to your model file..",default=os.path.join(current_dir,"model","m.h5"))
    ap.add_argument("--gpu",required=False,default=False,help="Flag to use GPU for predicting",type=str2bool)
    ap.add_argument("-p","--period",required=False,default=15,help="Period of prediction",type=int)
    ap.add_argument("--scatter-plot",required=False,default=False,help="Use scatter point for plot",type=str2bool)
    ap.add_argument("--fill-plot",required=False,default=True,help="Fill area under load curve",type=str2bool)
    ap.add_argument("--daydelta",required=False,default=0,type=int,help="day delta")
    ap.add_argument("--date",required=False,default=datetime.datetime.today(),help="date to predict")
    ap.add_argument("--iteration-delay",required=False,default=0.5,type=float,help="predict iteration delay")
    ap.add_argument("--show-graph",required=False,default=True,type=str2bool,help="set this flag to show graph")
    args = vars(ap.parse_args())
    date = args["date"]
    if type(date) != datetime.datetime :
        date = datetime.datetime.strptime(date,"%Y-%m-%d")
    daydelta = args["daydelta"]
    fill_plot=args["fill_plot"]
    scatter_plot = args["scatter_plot"]
    period = args["period"]
    model_path = args["model"]
    use_gpu = (args["gpu"])
    if args["show_graph"]:
        print("SHOW GRAPH!!")
        # PLOT
        plot_process = mp.Process(target=plot_load,kwargs={"fill":fill_plot,"scatter":scatter_plot,"daydelta":daydelta,"date":date.date()})
        plot_process.start() # run plot as a new process...
        time.sleep(3.5) # wait until graph show up
    # PREDICTION
    pred = predictor(model_path=model_path,
                     use_gpu=use_gpu,
                     predict_period=period,
                     daydelta=daydelta,
                     dt=date)
    pred.iteration_delay = args["iteration_delay"]
    pred.run() # start prediction process
    
    