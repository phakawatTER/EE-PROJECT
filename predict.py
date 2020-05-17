from tensorflow.keras.models import load_model
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


SOLAR_PV = {
        "00:00:00":	0,
        "00:30:00":	0,
        "01:00:00":	0,
        "01:30:00":	0,
        "02:00:00":	0,
        "02:30:00":	0,
        "03:00:00":	0,
        "03:30:00":	0,
        "04:00:00":	0,
        "04:30:00":	0,
        "05:00:00":	0,
        "05:30:00":	0,
        "06:00:00":	0,
        "06:30:00":	0,
        "07:00:00":	0,
        "07:30:00":	0.047619048,
        "08:00:00":	0.301587302,
        "08:30:00":	0.698412698,
        "09:00:00":	1.388888889,
        "09:30:00":	2.23015873,
        "10:00:00":	3.325396825,
        "10:30:00":	4.611111111,
        "11:00:00":	4.912698413,
        "11:30:00":	5.404761905,
        "12:00:00":	3.873015873,
        "12:30:00":	4.071428571,
        "13:00:00":	0.992063492,
        "13:30:00":	7.53968254,
        "14:00:00":	9.174603175,
        "14:30:00":	8.380952381,
        "15:00:00":	5.555555556,
        "15:30:00":	4.761904762,
        "16:00:00":	1.587301587,
        "16:30:00":	0.349206349,
        "17:00:00":	0.047619048,
        "17:30:00":	0,
        "18:00:00":	0,
        "18:30:00":	0,
        "19:00:00":	0,
        "19:30:00":	0,
        "20:00:00":	0,
        "20:30:00":	0,
        "21:00:00":	0,
        "21:30:00":	0,
        "22:00:00":	0,
        "22:30:00":	0,
        "23:00:00":	0,
        "23:30:00":	0
    }


class predictor:

    # Operation Cost
    BIOMASS_COST = 0.51  # Baht / kWH
    BIOGAS_COST = 1.2  # Baht / kWH
    SOLAR_COST = 0.03  # Baht / kWH
    OFFSET_FACTOR = 1.1
    

    def __init__(self, dt=datetime.datetime.today(), model_path="", predict_period=15, use_gpu=True, daydelta=0, message_callback=None, adjust_callback=None):
        #  Callback functions
        self.message_callback = message_callback  # callback function to alert event
        # callback function to adjust specific value
        self.adjust_callback = adjust_callback
        # Generatingself
        global SOLAR_PV
        self.BIOMASS_PV = 12.35  # kWH
        self.BIOGAS_PV = 14  # kWH
        self.SOLAR_PV = SOLAR_PV
        self.num_biomass = 3
        self.num_biogas = 2
        self.num_solar = 2
        if not os.path.exists(os.path.join(current_dir, "graph_data")):
            os.mkdir(os.path.join(current_dir, "graph_data"))
        self.should_terminate = False
        self.start_time = time.time()  # start prediction time
        self.dataframe = pd.DataFrame(columns=["time", "load"])
        self.predict_period = math.ceil(predict_period)
        self.iteration_delay = 0.1
        timedelta = datetime.timedelta(days=daydelta)
        dt = dt+timedelta  # get today datetime
        self.date = dt.date()
        self.start_datetime = dt.replace(
            hour=0, minute=0, second=0, microsecond=0)
        self.start_day = self.start_datetime.day
        self.use_gpu = use_gpu

        self.current_load = 0
        self.optimal_data = None

        self.device = "/device:CPU:0"
        if self.use_gpu:
            self.device = "/GPU:0"
        with tf.device(self.device):
            self.model = load_model(model_path)  # model for prediction

    def cal_optimal_cost(self, num_biomass=3, num_biogas=2, num_solar=2):
        time_portion = self.predict_period
        load = self.current_load * predictor.OFFSET_FACTOR  # new load with offset
        current_time = self.start_datetime.time() # current time being predicted
        current_time = time.strptime(str(current_time),"%H:%M:%S")
        solar_pv = None
        try:
            solar_pv = self.SOLAR_PV[current_time]
        except:
            before = None
            after = None
            isBefore=False
            isAfter=False
            for timestamp in self.SOLAR_PV:
                if isBefore and isAfter:
                    break
                t = time.strptime(timestamp,"%H:%M:%S")
                if t > current_time:
                    isBefore=True
                    before = timestamp
                else:
                    isAfter = True
                    after = timestamp
            value_before = self.SOLAR_PV[before]
            value_after = self.SOLAR_PV[after]
            solar_pv = value_before + value_after
            solar_pv = solar_pv / 2

        operation_cost = 0
        all_cost = {}
        for i in range(1,time_portion+1):
            biomass_portion = i
            for j in range(1,time_portion+1):
                solar_portion = j
                for k in range(1,time_portion+1):
                    biogas_portion = k
                    # PRODUCTS
                    biomass_load = num_biomass * \
                        (biomass_portion/time_portion) * \
                        self.predict_period/60*self.BIOMASS_PV  # kW
                    biogas_load = num_biogas * \
                        (biogas_portion/time_portion) * \
                        self.predict_period/60*self.BIOGAS_PV  # kW
                    solar_load = num_solar * \
                        (solar_portion/time_portion) * \
                        self.predict_period/60*solar_pv  # kW
                    # COST
                    biomass_cost = biomass_load * self.BIOMASS_COST
                    biogas_cost = biogas_load * self.BIOGAS_COST
                    solar_cost = solar_load * self.SOLAR_COST

                    total_load = biomass_load + biogas_load + solar_load
                    total_cost = biomass_cost + biogas_cost + solar_cost

                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,
                                                    biogas_portion, solar_portion)] = {}
                    all_cost["bm{}-bg{}-s{}".format(biomass_portion,
                                                    biogas_portion, solar_portion)]["timestamp"] = str(self.start_datetime.date()) 
                    all_cost["bm{}-bg{}-s{}".format(
                        biomass_portion, biogas_portion, solar_portion)]["bm_cost"] = biomass_cost
                    all_cost["bm{}-bg{}-s{}".format(
                        biomass_portion, biogas_portion, solar_portion)]["bg_cost"] = biogas_cost
                    all_cost["bm{}-bg{}-s{}".format(
                        biomass_portion, biogas_portion, solar_portion)]["s_cost"] = solar_cost
                    all_cost["bm{}-bg{}-s{}".format(
                        biomass_portion, biogas_portion, solar_portion)]["total_cost"] = total_cost
                    all_cost["bm{}-bg{}-s{}".format(
                        biomass_portion, biogas_portion, solar_portion)]["load"] = total_load

        optimal_solution = None
        max_load_solution = None
        prev_optimal_cost = None
        max_load = 0
        
        
        for key in all_cost:
            # print(key , all_cost[key])
            cost = all_cost[key]["total_cost"]
            _load_ = all_cost[key]["load"]
            if _load_ >= max_load:
                max_load = _load_
                max_load_solution = all_cost[key]
            if prev_optimal_cost == None:
                if _load_ < load:
                    continue
                prev_optimal_cost = cost
                optimal_solution = all_cost[key]
            elif cost <= prev_optimal_cost:
                if _load_ < load:
                    continue
                prev_optimal_cosprev_optimal_costt = cost
                optimal_solution = all_cost[key]

        print(optimal_solution, prev_optimal_cost, max_load)
        if optimal_solution == None:
            try:
                self.message_callback("Power Error!", "Cannot Produce Enough Power!\
                                      Try to increase number of generator or generating power.\
                                      Expected {} kW but produced atmost {} kW. \
                                      ".format(round(load, 2), round(max_load, 2)))
            except Exception as err:
                print(err)
                pass
            
        if optimal_solution == None:
            return max_load_solution
        return optimal_solution

    def get_predict_datetime(self, prediction_time):
        day = (prediction_time.weekday()+1) % 7
        month = prediction_time.month
        hour = prediction_time.hour
        minute = prediction_time.minute
        return np.array([[day, month, hour, minute]]).astype(np.float32).reshape(-1, 4)

    def predict_load(self, time):
        with tf.device(self.device):
            p_load = self.model.predict(time)
            return p_load

    def task(self):
        weekday_index_in_month = find_weekday_order_number(self.start_datetime)
        timedelta = datetime.timedelta(minutes=self.predict_period)
        self.start_datetime += timedelta
        current_day = self.start_datetime.day
        day = (self.start_datetime.weekday()+1) % 7
        month = self.start_datetime.month
        hour = self.start_datetime.hour
        minute = self.start_datetime.minute
        input_vector = [day, weekday_index_in_month, month, hour, minute]
        tensor = np.array([input_vector]).astype(
            np.float32).reshape(-1, len(input_vector))  # input tensor
        load = self.predict_load(tensor)  # predict load
        self.current_load = load[0][0]  # current predicted load
        opt_sol = self.cal_optimal_cost(
            num_biomass=self.num_biomass,
            num_biogas=self.num_biogas,
            num_solar=self.num_solar
        )  # calculate optimal cost with the 2 generators
        opt_load = opt_sol["load"]  # optimal load
        opt_cost = opt_sol["total_cost"]  # optimal cost
        opt_solar_cost = opt_sol["s_cost"]  # optimal cost for solar
        opt_biomass_cost = opt_sol["bm_cost"]  # optimal cost for biomass
        opt_biogas_cost = opt_sol["bg_cost"]  # optimal cost for biogas
        df = pd.DataFrame(data={
            "time": [self.start_datetime],
            "load": [load[0][0]],
            "cost": [opt_cost],
            "opt_load":[opt_load],
            "solar_cost": [opt_solar_cost],
            "biomass_cost": [opt_biomass_cost],
            "biogas_cost": [opt_biogas_cost]
        })
        print(df)
        self.dataframe = self.dataframe.append(df, ignore_index=True)
        self.dataframe.to_csv(os.path.join(
            current_dir, "graph_data", "load_{}.csv".format(self.date)))

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


if __name__ == "__main__":
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
    ap.add_argument("-m", "--model", required=False, help="Path to your model file..",
                    default=os.path.join(current_dir, "model", "m.h5"))
    ap.add_argument("--gpu", required=False, default=False,
                    help="Flag to use GPU for predicting", type=str2bool)
    ap.add_argument("-p", "--period", required=False, default=15,
                    help="Period of prediction", type=int)
    ap.add_argument("--scatter-plot", required=False, default=False,
                    help="Use scatter point for plot", type=str2bool)
    ap.add_argument("--fill-plot", required=False, default=True,
                    help="Fill area under load curve", type=str2bool)
    ap.add_argument("--daydelta", required=False,
                    default=0, type=int, help="day delta")
    ap.add_argument("--date", required=False,
                    default=datetime.datetime.today(), help="date to predict")
    ap.add_argument("--iteration-delay", required=False,
                    default=0.5, type=float, help="predict iteration delay")
    ap.add_argument("--show-graph", required=False, default=True,
                    type=str2bool, help="set this flag to show graph")
    args = vars(ap.parse_args())
    date = args["date"]
    if type(date) != datetime.datetime:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    daydelta = args["daydelta"]
    fill_plot = args["fill_plot"]
    scatter_plot = args["scatter_plot"]
    period = args["period"]
    model_path = args["model"]
    use_gpu = (args["gpu"])
    if args["show_graph"]:
        print("SHOW GRAPH!!")
        # PLOT
        plot_process = mp.Process(target=plot_load, kwargs={
                                  "fill": fill_plot, "scatter": scatter_plot, "daydelta": daydelta, "date": date.date()})
        plot_process.start()  # run plot as a new process...
        time.sleep(3.5)  # wait until graph show up
    # PREDICTION
    pred = predictor(model_path=model_path,
                     use_gpu=use_gpu,
                     predict_period=period,
                     daydelta=daydelta,
                     dt=date)
    pred.iteration_delay = args["iteration_delay"]
    pred.run()  # start prediction process
