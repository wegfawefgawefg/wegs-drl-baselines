threshold = 5 #years

def millis_to_years(millis):
    seconds = millis / 1000
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    years = days / 365

    return years

years = 0
doublings = 0
start = 1
for i in range(32):
    print(start)
    start *= 2
    years = millis_to_years(start)
    if years > threshold:
        break
    doublings += 1

print(years)
print(start)
print(doublings)