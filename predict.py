from  tensorflow.keras.models import load_model
import numpy as np
import schedule
import datetime 
import pandas as pd
import time
import os


current_dir = os.path.dirname(__file__)

class predictor:
    # predict_period is in "minute"
    def __init__(self,model_path="",predict_period=0.05,test_mode=False ):
        self.test_mode = test_mode
        self.start_time = time.time() # start prediction time
        self.dataframe = pd.DataFrame(columns=["time","load"])
        self.predict_period = predict_period
        self.model = load_model(model_path) # model for prediction

        today = datetime.datetime.today()
        self.day = (today.weekday()+1)%7
        self.year = today.year
        self.month = today.month
        self.hour = 0
        self.minute = 0

    def get_predict_datetime(self):
        today = datetime.datetime.today()
        day = (today.weekday()+1)%7
        month = today.month
        time = str(datetime.datetime.now()).split(" ")[1]
        hour = time.split(":")[0]
        minute = time.split(":")[1]
        return np.array([[day,month,hour,minute]]).astype(np.float32).reshape(-1,4)

    def predict_load(self,time):
        p_load = self.model.predict(time)
        return p_load

    def job(self):
        input_time = self.get_predict_datetime()
        load = self.predict_load(input_time)
        time = datetime.datetime.now()
        df = pd.DataFrame(data={"time":[time],"load":[load[0][0]]})
        self.dataframe = self.dataframe.append(df,ignore_index=True)
        self.dataframe.to_csv(os.path.join(current_dir,"graph_data","load.csv"))

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
                time.sleep(self.predict_period*60)
            else:
                self.test()
                time.sleep(0.0001)
        print(self.dataframe)

if __name__=="__main__":
    import sys
    import argparse 
    import multiprocessing as mp
    
    plot_process = mp.Process(target=os.system,args=("python {}".format(os.path.join(current_dir,"plot_load.py")),) )
    plot_process.start() # run plot as a new process...
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model",required=False,help="Path to your model file..",default="model/model1.h5")
    ap.add_argument("-t","--test",required=False,help="Enable test mode",default=False,type=bool)
    args = vars(ap.parse_args())
    model_path = args["model"]
    test_mode = args["test"]
    pred = predictor(model_path=model_path,test_mode=test_mode)
    pred.run() # start prediction process
    
    



