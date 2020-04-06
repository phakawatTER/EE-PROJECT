import pandas as pd
from matplotlib import pyplot as plt
import os
DAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
TIMES = []
ENV_NAME = "ee-env"
dataframe = pd.DataFrame(columns=['time', 'load'])
current_dir = os.path.dirname(__file__)  # current path of script file
dataset_path = os.path.join(current_dir,"dataset")
# list directory of each day dataset
directory = [dir for dir in os.listdir(dataset_path) if not os.path.isfile(dir)
             and dir != ENV_NAME]


def get_day(day):
    if day in DAYS:
        return day
    else:
        for d in DAYS:
            if d in day:
                return d


def get_month(month):
    if month in MONTHS:
        return month
    else:
        for d in MONTHS:
            if d in month:
                return d
        print(month)


for d in directory:
    year,_,_,_ = d.split("_")
    files = os.listdir(os.path.join(dataset_path,d))
    try:
        for file in files:
            try:
                try:
                    dd, mm, _, _ = file.split("_")
                except:
                    dd,mm,_ =file.split("_")
                weekday = dd.lower()
                month = mm.lower()
                dd = get_day(weekday)  # lowercase day
                order_in_month = weekday.replace(dd,"")
                if order_in_month == "":order_in_month=0
                order_in_month = int(order_in_month)
                mm = get_month(month)  # lowercase month
                dd = DAYS.index(dd)
                mm = MONTHS.index(mm)
                dt = "{} {} ".format(dd, mm)
                file_path = os.path.join(dataset_path,d ,file)
                csv = pd.read_csv(file_path)
                csv.rename(columns={"Category": "time",
                                    "Krabi": "load"}, inplace=True)
                for i, t in enumerate(csv["time"]):
                    try:
                        new_time = dt+t
                        csv.at[i, "time"]= new_time
                        csv.at[i,"year"]=year
                        csv.at[i,"order_in_month"]=order_in_month
                    except:
                        pass
                dataframe = dataframe.append(csv, ignore_index=True)
            except:
                pass
    except:
        pass
print(dataframe)
clusters = {}
def format_date(x):
    if len(str(x)) == 1:
        x= "0"+str(x)
    return x
def format_time(time):
    hour,minute = time.split(":")
    second = "00"
    if len(hour) == 1:
        hour = "0"+hour
    return  ":".join([hour,minute,second])

for i, time in enumerate(dataframe["time"]):

    try:
        day, month, time = time.split(" ")
        hour, minute = time.split(":")
        cluster = "{}-{}".format(day, month)
        tt = format_time(time)
        dd = format_date(int(day)+1)
        mm = format_date(int(month)+1)
        
        norm_time = (int(day)+int(month)+int(hour)+int(minute))/(6+11+23+59)
        date = f"2016-{mm}-{dd} {tt}"
        dataframe.at[i, "date"]=date
        dataframe.at[i, "time"]=norm_time
        dataframe.at[i, "day"]= day
        dataframe.at[i, "month"]= month
        dataframe.at[i, "hour"]= int(hour)
        dataframe.at[i, "minute"]= int(minute)
        if cluster not in clusters:
            clusters[cluster] = {"time": [], "load": []}
        clusters[cluster]["time"].append(dataframe["time"][i])
        clusters[cluster]["load"].append(dataframe["load"][i])
    except Exception as err:
        pass
    
dataframe.drop(columns=["time"],axis=1)

dataframe= dataframe.sort_values(by="date",ascending=True)
# dataframe=dataframe.reindex(columns=["norm_time","load"])
dataframe=dataframe.reindex(columns=["day","order_in_month","month","hour","minute","load"])
# dataframe=dataframe.reindex(columns=["date","load"])
dataframe=dataframe.dropna()# drop nan row
dataframe.to_csv("training_set.csv", index=False)

print("Successfully create training set ...")
