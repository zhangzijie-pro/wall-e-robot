a = [{"Alice":0.7}, {"Bob":0.3}]

names = [k for name in a for k,_ in name.items() ]

print(names)