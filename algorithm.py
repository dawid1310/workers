import numpy as np
import random as ran
import json
import geopy.distance
import datetime


with open('conf.json') as json_file:
    configuration = json.load(json_file)

time_perspective=configuration["time_perspective"]
hourperday=configuration["hourperday"]
w_cost = configuration["w_cost"]
w_dist = configuration["w_dist"]
norm_cost = configuration["norm_cost"]
norm_dist = configuration["norm_dist"]
pop_size = configuration["pop_size"]
prop_cross = configuration["prop_cross"]
prop_mut = configuration["prop_mut"]
iterations = configuration["iterations"]

def give_date(min_date_start):
    return datetime.date.fromisoformat(min_date_start[0:10])

def distance(coords_1, coords_2):
    return geopy.distance.geodesic(coords_1, coords_2).km

def rand_assign(row,task_workers):
    if len(task_workers)>0:
        row=np.zeros(len(row))
        w = ran.randint(0, len(task_workers)-1 )
        row[task_workers[w]] = 1

    return row

def tasks_workers_competences(workers, tasks):
    tasks_workers = []
    tasks_start = []
    i=0
    today = datetime.date.today()
    for t in tasks:
        r=[]
        tasks_workers.append(r)
        tasks_start.append(0)
        if give_date(t['min_date_start']) <= today + datetime.timedelta(time_perspective):
            j=0
            for w in workers:
                if not False in np.in1d(t['competences_required'],w['competences']):
                    if distance(t['location'],w['location'])<w['max_dist']:
                        tasks_workers[i].append(j)
                j=j+1
            if len(tasks_workers[i])==0:
                tasks_start[i]=-2 # -2 means no workers with apropriate competences or distance
            else:
                tasks_start[i]=give_date(t['min_date_start'])
        else:
            tasks_start[i]=-1 #-1 means min_date_start after time_perspective
        i=i+1
    return tasks_workers, tasks_start

def solution_to_json(sol, tasks, workers,tasks_start):
    sol_dict=[]
    for i in range(len(sol)):
        if max(sol[i]==1):
            sol_dict.append("task_id: "+str(tasks[i]['id'])+",worker_id:"+str(workers[np.where(sol[i]==1)[0][0]]['id'])+",date_start:"+str(tasks_start[i]))
        elif tasks_start[i]==-1:
            sol_dict.append("task_id: " + str(tasks[i]['id']) + ",info: 'min_date_start after time_perspective'")
        elif tasks_start[i] == -2:
            sol_dict.append("task_id: " + str(tasks[i]['id']) + ",info: 'no workers with apropriate competences or distance'")

    with open("result.json", "w") as outfile:
        json.dump(sol_dict, outfile)

def cost_min(plan):
    cost=0
    row_num=0
    for row in plan:
        if np.where(row == 1)[0]:
            cost=cost+workers[np.where(row==1)[0][0]]['salary_per_hour']*tasks[row_num]['time']
        row_num = row_num + 1
    return cost

def distance_min(plan):
    dist = 0
    row_num = 0
    for row in plan:
        if np.where(row == 1)[0]:
            dist = dist + distance(workers[np.where(row == 1)[0][0]]['location'], tasks[row_num]['location'])
        row_num = row_num + 1
    return dist

#def delay_min(tasks_start_sol, tasks_start):
#    delay=0
#    for i in range(0,len(tasks_start_sol)):
#        print(tasks_start_sol[i])
#        print(days)
#    return(delay)

def evaluation(sol, tasks_start_sol, tasks_start):
     return w_cost*norm_cost*cost_min(sol)+w_dist*norm_dist*distance_min(sol)

def cross(sol1, sol2):
    point = ran.randint(0, len(sol1) - 1)

    for i in range(0,point):
        sol1_work_position = np.where(sol1==1)[0][0]
        sol2_work_position = np.where(sol2==1)[0][0]
        np.zeros(sol1,sol2)
        sol1[sol2_work_position]=1
        sol2[sol1_work_position]=1

def repair(plan, tasks_start, tasks, workers, ts):
    workers_duty = []

    for i in range(0,len(workers)):
        workers_duty.append([])
        for j in range(0,len(tasks)):
            if plan[j][i]==1:
                workers_duty[i].append([j,[ts[j],tasks[j]['time']]])

        if len(workers_duty[i])>1:
            w_duty=sorted(workers_duty[i], key=lambda workers: workers[1])
            for k in range(1,len(w_duty)):
                if w_duty[k][1][0]<w_duty[k-1][1][0]+datetime.timedelta(w_duty[k-1][1][1]/hourperday):
                    tasks_start[w_duty[k][0]]=ts[w_duty[k][0]]+datetime.timedelta(w_duty[k-1][1][1]/hourperday)
    return tasks_start

with open('workers.json') as json_file:
    workers = json.load(json_file)

with open('tasks.json') as json_file:
    tasks = json.load(json_file)


task_workers, tasks_start=tasks_workers_competences(workers, tasks)
work_num = len(workers)
tasks_num = len(tasks)

solutions=[]
tasks_start_pop=[]

for i in range(pop_size):
    sol = np.zeros((tasks_num, work_num))
    for j in range(tasks_num):
        sol[j] = rand_assign(sol[j], task_workers[j])
    solutions.append(sol)
    tasks_start_pop.append(repair(sol,tasks_start,tasks,workers, tasks_start))

best_ever=100000000000
for it in range(iterations):
    min_cost=1000000000
    min_sol=-1
    costs=[]
    costs_grow=[]
    costs_sum=0
    new_population=[]
    #SELECTION - COST
    for i in range(pop_size):
        c=evaluation(solutions[i], tasks_start_pop[i], tasks_start)
        costs.append(c)
        costs_sum=costs_sum+c
        costs_grow.append(costs_sum)

    for i in range(len(costs_grow)-1):
        costs_grow[i]=costs_sum-costs_grow[i]
    costs_grow.sort()

    costs_grow_np = np.array(costs_grow)

    for i in range(pop_size):
        r = ran.randint(1, round(costs_sum))
        new_population.append(solutions[np.nonzero(costs_grow_np > r)[0][0]])

    #crossover

    to_cross=[]

    for i in range(pop_size):
        if ran.random()<prop_cross:
            to_cross.append(i)

    if len(to_cross) % 2 != 0:
        to_cross.pop(ran.randint(0,len(to_cross)-1))

    while len(to_cross)>1:
        r=ran.randint(0,len(to_cross)-1)
        index1=to_cross[r]
        to_cross.pop(r)
        r = ran.randint(0, len(to_cross) - 1)
        index2 = to_cross[r]
        to_cross.pop(r)

        cross_point=ran.randint(1,len(new_population[0])-1)

        for i in range(0,cross_point):
            if max(new_population[index1][i]) == 1:
                in1=np.where(new_population[index1][i] == 1)[0][0]
                in2=np.where(new_population[index2][i] == 1)[0][0]
                new_population[index1][i][in1] = 0
                new_population[index1][i][in2] = 1

                new_population[index2][i][in2] = 0
                new_population[index2][i][in1] = 1


    #MUTATION
    for i in range(pop_size):
        if ran.random()<prop_mut:
            j=ran.randint(0,tasks_num-1)
            sol=new_population[i][j].copy()
            new_population[i][j] = rand_assign(sol, task_workers[j])


    solutions=[]
    solutions=new_population.copy()

    costs=[]
    for i in range(pop_size):
        c = evaluation(solutions[i], tasks_start_pop[i], tasks_start)
        costs.append(c)
        tasks_start_pop[i]=repair(solutions[i],tasks_start_pop[i], tasks, workers, tasks_start)
    index_min = np.argmin(costs)
    if min(costs)<best_ever:
        best_ever=min(costs)
        best_sol=solutions[index_min].copy()
    print(best_ever)

print(best_sol)
solution_to_json(best_sol, tasks, workers, tasks_start)


