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
from plot_load import plot_load

# set keras backend
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

current_dir = os.path.dirname(__file__)

class predictor:
    # predict_period is in "minute"
    def __init__(self,dt=datetime.datetime.today(),model_path="",predict_period=15,use_gpu=True,daydelta=0):
        if not os.path.exists(os.path.join(current_dir,"graph_data")):
            os.mkdir(os.path.join(current_dir,"graph_data"))
        self.should_terminate = False
        self.start_time = time.time() # start prediction time
        self.dataframe = pd.DataFrame(columns=["time","load"])
        self.predict_period = math.ceil(predict_period)
        timedelta = datetime.timedelta(days=daydelta)
        today = dt+timedelta # get today datetime
        self.date = today.date()+timedelta
        self.start_datetime = today.replace(hour=0,minute=0,second=0,microsecond=0)
        self.start_day = self.start_datetime.day
        self.use_gpu = use_gpu
        self.device = "/device:CPU:0"
        if self.use_gpu:
            self.device = "/GPU:0"
        with tf.device(self.device):
            self.model = load_model(model_path) # model for prediction
                        
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
        timedelta = datetime.timedelta(minutes=self.predict_period)
        self.start_datetime += timedelta
        current_day = self.start_datetime.day
        day = (self.start_datetime.weekday()+1)%7
        month = self.start_datetime.month
        hour = self.start_datetime.hour
        minute = self.start_datetime.minute
        tensor = np.array([[day,month,hour,minute]]).astype(np.float32).reshape(-1,4)
        load = self.predict_load(tensor)
        df = pd.DataFrame(data={"time":[self.start_datetime],"load":[load[0][0]]})
        self.dataframe=self.dataframe.append(df,ignore_index=True)
        self.dataframe.to_csv(os.path.join(current_dir,"graph_data","load_{}.csv".format(self.date)))
        
        if self.start_day != current_day:
            self.should_terminate = True
        print(f"{self.start_datetime} ----> {load[0][0]}")

    def run(self):
        while True:
            if not self.should_terminate:
                self.task()
            else:
                break
        print(f"End Prediction on {datetime.date.today()}")

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
    
    manager = mp.Manager()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model",required=False,help="Path to your model file..",default=os.path.join(current_dir,"model","model1.h5"))
    ap.add_argument("--gpu",required=False,default=False,help="Flag to use GPU for predicting",type=str2bool)
    ap.add_argument("-p","--period",required=False,default=15,help="Period of prediction",type=int)
    ap.add_argument("--scatter-plot",required=False,default=False,help="Use scatter point for plot",type=str2bool)
    ap.add_argument("--fill-plot",required=False,default=True,help="Fill area under load curve",type=str2bool)
    ap.add_argument("--daydelta",required=False,default=0,type=int,help="day delta")
    ap.add_argument("--date",required=False,default=datetime.datetime.today(),help="date to predict")
    args = vars(ap.parse_args())
    date = args["date"]
    if date :
        date = datetime.datetime.strptime(date,"%Y-%m-%d")
    daydelta = args["daydelta"]
    fill_plot=args["fill_plot"]
    scatter_plot = args["scatter_plot"]
    period = args["period"]
    model_path = args["model"]
    use_gpu = (args["gpu"])
    
    # PLOT
    plot_process = mp.Process(target=plot_load,kwargs={"fill":fill_plot,"scatter":scatter_plot,"daydelta":daydelta,"date":date.date()})
    plot_process.start() # run plot as a new process...
    
    # PREDICTION
    pred = predictor(model_path=model_path,
                     use_gpu=use_gpu,
                     predict_period=period,daydelta=daydelta,dt=date)
    pred.run() # start prediction process
    
    