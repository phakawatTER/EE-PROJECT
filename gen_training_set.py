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
# list directory of each day dataset
directory = [dir for dir in os.listdir() if not os.path.isfile(dir)
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
    files = os.listdir(d)
    try:
        for file in files:
            try:
                try:
                    dd, mm, _, _ = file.split("_")
                except:
                    dd,mm,_ =file.split("_")
                dd = get_day(dd.lower())  # lowercase day
                mm = get_month(mm.lower())  # lowercase month
                dd = DAYS.index(dd)
                mm = MONTHS.index(mm)
                dt = "{} {} ".format(dd, mm)
                file_path = os.path.join(d, file)
                csv = pd.read_csv(file_path)
                csv.rename(columns={"Category": "time",
                                    "Krabi": "load"}, inplace=True)
                for i, t in enumerate(csv["time"]):
                    try:
                        new_time = dt+t
                        csv.set_value(i, "time", new_time)
                    except:
                        pass
                dataframe = dataframe.append(csv, ignore_index=True)
            except:
                pass
    except:
        pass
clusters = {}
for i, time in enumerate(dataframe["time"]):

    dd = 24
    mm = 60
    ss = 60
    try:
        day, month, time = time.split(" ")
        hour, minute = time.split(":")
        cluster = "{}-{}".format(day, month)
        date = f"2016-{int(month)+1}-{int(day)+1}"
        dataframe.set_value(i, "date",date )
        # dataframe.set_value(i, "day", int(day))
        # dataframe.set_value(i, "month", int(month))
        dataframe.set_value(i, "hour", int(hour))
        dataframe.set_value(i, "minute", int(minute))
        if cluster not in clusters:
            clusters[cluster] = {"time": [], "load": []}
        clusters[cluster]["time"].append(dataframe["time"][i])
        clusters[cluster]["load"].append(dataframe["load"][i])
    except Exception as err:
        print(err)
        pass
dataframe.drop(columns=["time"],axis=1)
dataframe=dataframe.reindex(columns=["date","hour","minute","load"])
dataframe.to_csv("training_set.csv", index=False)

# color_list = ["ro", "go", "bo", "yo"]
# for i, c in enumerate(clusters):
#     cluster = clusters[c]
#     time = cluster["time"]
#     load = cluster["load"]
#     plt.plot(time, load, color_list[i % len(color_list)-1])
# plt.show()
