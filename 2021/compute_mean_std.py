import statistics





# initializing list
test_list = [0.34684722809987306,
0.3487906145898711,
0.3465731050753131]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
