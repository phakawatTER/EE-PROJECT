
import datetime 

def find_weekday_order_number(date):
    timedelta = datetime.timedelta(days=7)
    day = date.weekday()
    month = date.month
    year = date.year
    count = 0
    while True:
        date = date - timedelta
        if date.month != month:
            break
        count+=1
    return count


if __name__=="__main__":
    timedelta = datetime.timedelta(days=1+14)
    date = datetime.datetime.now() + timedelta  
    print(date)
    find_weekday_order_number(date)