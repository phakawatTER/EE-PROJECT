from  tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import numpy as np
import schedule
import datetime 
import pandas as pd
import time
import os
import math

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

current_dir = os.path.dirname(__file__)

class predictor:
    # predict_period is in "minute"
    def __init__(self,model_path="",predict_period=15,test_mode=False,use_gpu=True):
        self.test_mode = test_mode
        self.start_time = time.time() # start prediction time
        self.dataframe = pd.DataFrame(columns=["time","load"])
        self.predict_period = math.ceil(predict_period)
        self.iteration = 0
        today = datetime.datetime.today() # get today datetime
        self.start_datetime = today.replace(hour=0,minute=0,second=0.00)
        self.use_gpu = use_gpu
        self.device = "/device:CPU:0"
        if self.use_gpu:
            self.device = "/GPU:0"
        with tf.device(self.device):
            self.model = load_model(model_path) # model for prediction
            
        # variable for testing
        today = datetime.datetime.today()
        self.day = (today.weekday()+1)%7
        self.year = today.year
        self.month = today.month
        self.hour = 0
        self.minute = 0

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

    def job(self): # function predict every 15 minutes
        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second
        multiplier = math.ceil(minute / self.predict_period)
        if multiplier == 0:
            multiplier +=1
        next_predict_minute = (self.predict_period * multiplier) % 60
        minute_diff = next_predict_minute - minute 
        if minute_diff == 0 :
            minute_diff += self.predict_period  
        timedelta = datetime.timedelta(minutes=minute_diff)
        prediction_time = now+timedelta
        sleep_time = minute_diff*60
        input_time = self.get_predict_datetime(prediction_time)
        load = self.predict_load(input_time)
        df = pd.DataFrame(data={"time":[prediction_time],"load":[load[0][0]]})
        self.dataframe = self.dataframe.append(df,ignore_index=True)
        self.dataframe.to_csv(os.path.join(current_dir,"graph_data","load.csv"))
        time.sleep(sleep_time-int(second))
        

    def test(self): # for testing
        day = self.day
        month =self.month
        hour = self.hour
        minute = self.minute
        self.minute += 1
        if self.minute % 60 == 0:
            self.minute = 0
            self.hour+=1
        if self.hour % 24 ==0 :
            self.hour =0 
        _input = np.array([[day,month,hour,minute]]).astype(np.float32).reshape(-1,4)
        print(_input)
        load = self.predict_load(_input)
        _time = time.time()
        df = pd.DataFrame(data={"time":[_time],"load":[load[0][0]]})
        self.dataframe=self.dataframe.append(df,ignore_index=True)
        self.dataframe.to_csv(os.path.join(current_dir,"graph_data","load.csv"))


    def run(self):
        while True:
            schedule.run_pending()
            if not test_mode:
                self.job()
            else:
                self.test()
                time.sleep(1)
        print(self.dataframe)

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
    plot_process = mp.Process(target=os.system,args=("python {}".format(os.path.join(current_dir,"plot_load.py")),) )
    plot_process.start() # run plot as a new process...
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model",required=False,help="Path to your model file..",default="model/model1.h5")
    ap.add_argument("-t","--test",required=False,help="Enable test mode",default=False,type=str2bool)
    ap.add_argument("--gpu",required=False,default=False,help="Flag to use GPU for predicting",type=str2bool)
    ap.add_argument("-p","--period",required=False,default=15,help="Period of prediction")
    args = vars(ap.parse_args())
    period = args["period"]
    model_path = args["model"]
    test_mode = (args["test"])
    use_gpu = (args["gpu"])
    print(use_gpu, test_mode)
    
    pred = predictor(model_path=model_path,
                     test_mode=test_mode,
                     use_gpu=use_gpu,
                     predict_period=period)
    pred.run() # start prediction process
    
    



