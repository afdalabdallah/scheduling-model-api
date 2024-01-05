# from __future__ import print_function, division
import numpy as np
import random

# First number means # day of the week
# The rest is the session of the day
# Ex. 301 = Wednesday, Session 1
# Ex. 110 = Monday, Session 10
# A session means an hour, Session 1 starts from 7.00
# That means Sesssion 2 is 8.00, Session 3 is 9.00, and so on.
default_sesi = []

# Hanya perlu inisiasi satu kali di awal karena prefrensi dosen tidak akan beruabh
# selama algoritma berjalan, jadi hanya menggunakan 1 variabel saja untuk pengecekan 
# prefrensi
# Struktur {"DA" : ["101,"102"]}
# key = kode dosen, value: list sesi prefrensi
dosenPrefensiDict = {}
for i in range(1, 6):
    for j in range(1, 10, 2):
        temp_sesi = ""
        if j < 10:
            default_sesi.append(str(i) + "0" + str(j))
        else:
            default_sesi.append(str(i) + str(j))
all_sesi = []
for i in range(1, 6):
    for j in range(1, 11):
        temp_sesi = ""
        if j < 10:
            all_sesi.append(str(i) + "0" + str(j))
        else:
            all_sesi.append(str(i) + str(j))


class GeneticAlgorithm():
    """An implementation of a Genetic Algorithm which will try to produce the user
    specified target string.

    Parameters:
    -----------
    population_size: int
        The number of individuals (possible solutions) in the population.
    """

    def __init__(self, population_size,raw_data):
        self.population_size = population_size
        self.data = {}
        self.sesi = []
        self.transferred = {}
        self.skpb_sesi = []
        # f = open('test.json')
        self.data = raw_data
        self.unwanted_sesi = self.data["unwanted_sesi"]
        self.data_ruangan = self.data["ruangan"]
        self.constraintViolated = {}
        # f.close()

    def preferensiToSesi(self, preferensiObj):
        hariDict = {
            "Senin": "1",
            "Selasa": "2",
            "Rabu": "3",
            "Kamis": "4",
            "Jumat": "5"
        }
        sesiDict = {
            "Sesi 1": "01",
            "Sesi 2": "02",
            "Sesi 3": "03",
            "Sesi 4": "04",
            "Sesi 5": "05",
            "Sesi 6": "06",
            "Sesi 7": "07",
            "Sesi 8": "08",
            "Sesi 9": "09",
            "Sesi 10": "10"
        }
        listPrefrensi = []
        tempSesi = ""
        
        for hari in preferensiObj['hari']:
         
            tempSesi = tempSesi + hariDict[hari]
            for sesi in preferensiObj['sesi']:
                listPrefrensi.append(tempSesi + sesiDict[sesi])
            tempSesi = ""

        return listPrefrensi

    def _initIndividu(self):
        print(len(self.sesi))
        cols = len(all_sesi) 
        rows = len(self.data_ruangan)
        timetable = [['' for j in range(cols)] for i in range(rows)]
      
        # Inisiasi untuk SKP+B
        self.timetable_skpb=['' for i in range(cols)]
        self.list_skpb = []
        for data in self.data['data']:
            individuSesi = self.preferensiToSesi(data["preferensi"])
            if data['dosen'] not in dosenPrefensiDict:
                dosenPrefensiDict[data['dosen']] = individuSesi

            if data['mata_kuliah'][0:2] != "UG":
                class_activity = data['dosen']+data['mata_kuliah']+data['kelas']
                random_rows = np.random.choice(np.arange(len(self.data_ruangan)), size=1, replace=False)
                while(True):
                    random_cols = np.random.choice(np.arange(len(all_sesi)), size=1, replace=False)
                    if (all_sesi[int(random_cols)] not in self.unwanted_sesi) and (timetable[int(random_rows)][int(random_cols)] == ''):
                        break
              
                
                class_activity = data['dosen']+data['mata_kuliah']+data['kelas']
                timetable[int(random_rows)][int(random_cols)] = class_activity 
                
            else: # Input data SKPB
                first_digit = int(data['sesi'][0]) - 1
                last_digit = int(data['sesi'][2]) - 1
                
                skpb_col = first_digit * 10 + last_digit
                self.timetable_skpb[skpb_col] = data['dosen']+data['mata_kuliah']+data['kelas']+data['sesi']
                self.list_skpb.append(data['dosen']+data['mata_kuliah']+data['kelas']+data['ruangan']+data['sesi'])
                self.skpb_sesi.append(data['sesi'])
        
        
        # list_perkuliahan sudah menjadi Individu
        return timetable
        
    def initTransferredActivity(self, reset):
        if reset:
            for key in self.transferred:
                self.transferred[key] = False
        else:
            individu = self.population[0]
            timeslot = len(individu[0])
            ruangan = len(individu)
            for room in range(ruangan):
                for time in range(timeslot):
                    activity = individu[room][time]
                    if(activity != ''):
                        self.transferred[activity] = False


    def removeUnwantedSesi(self):
        for unwanted in self.unwanted_sesi:
            if unwanted in default_sesi:
                default_sesi.remove(unwanted)
            if unwanted in all_sesi:
                all_sesi.remove(unwanted)
        return all_sesi

    def _initialize(self):
        """ Initialize population with random strings """
        self.population = []
        for _ in range(self.population_size):
            individual = self._initIndividu()
            # print(individual)
            self.population.append(individual)
        self.initTransferredActivity(False)

    # Return number of violation [x,y,z,p,q] for an individual
    def _individuConstrain(self, individu):
        x = 0 # First to third constraint
        y = 0
        z = 0
        p = 0 # fourth constraint
        q = 0 # prefrensi
        # print("INI DI INDIVIDU CONSTRAINT")
        # print(individu)
        timeslot = len(individu[0])
        ruangan = len(individu)
        count_perkuliahan_table = 0
        # print(timeslot, ruangan)
        # First constraint: dosen mengajar di waktu yang sama
        for time in range(timeslot):
            count_same_dosen = {}
            for room in range (ruangan):
                activity = individu[room][time]
                if(activity != ''):
                    dosen = activity[0:2]
                    count_same_dosen[dosen] =  count_same_dosen.get(dosen, 0) + 1
                    count_perkuliahan_table = count_perkuliahan_table + 1
            duplicate_in_timeslot = sum(count-1 for count in count_same_dosen.values() if count > 1 )
            x = x + duplicate_in_timeslot

        # Third constraint: dosen tidak mengajar lebih dari 2x dalam sehari
        for room in range(ruangan):
            count_dosen_mengajar = {}
            for time in range(timeslot):
                activity = individu[room][time]
                dosen = activity[0:2]
                # Per 10 sesi di 1 hari
                if(time % 10 == 0 and time != 0):
                    double_dosen_mengajar = sum(count-1 for count in count_dosen_mengajar.values() if count > 2 )
                    y = y + double_dosen_mengajar
                    count_dosen_mengajar = {}
                if(activity != ''):
                    count_dosen_mengajar[dosen] = count_dosen_mengajar.get(dosen,0) + 1
                    index_to_sesi = 101 + time // 10 * 100 + time % 10
                    # Fifth Constrain (Prefrensi)
                    if str(index_to_sesi) not in dosenPrefensiDict[dosen]:
                        q = q + 1
                    # Fourth Constraint (SKPB)
                    for s in range(len(self.list_skpb)):
                        if abs(index_to_sesi - int(self.list_skpb[s][12:15])) < 2:
                            p = p+1
        
        return [x,y,z,p,q]

    def individuFitness(self, individual):
        w1 = 45
        w2 = 45
        w3 = 9
        w4 = 1
        x,y,z,p,q = self._individuConstrain(individual)
        f1 = 0
        f2= 0
        f3=0
        f4 =0
        if(x == 0):
            f1 = w1
        else:
            f1 = w1/(x + 1e-8)

        if(y == 0):
            f2 = w2
        else:
            f2 = w2/(y + 1e-8)

        if(p == 0):
            f3 = w3
        else:
            f3 = w3/(p + 1e-8)
        
        if(q == 0):
            f4 = w4
        else:
            f4 = w4/(q + 1e-8)


        fitness = f1 + f2 +f3 + f4
        return [fitness, x,y,z,p,q]

    def _calculate_fitness(self):
        """ Calculates the fitness of each individual in the population """
        population_fitness = []
        w1 = 45
        w2 = 45
        w3 = 9
        w4 = 1
        
        for individual in self.population:
            # loss: Array[x,y,z,p,q]
            x,y,z,p,q = self._individuConstrain(individual)
            # print(x,y,z,p,q)
            # print(x)
            f1 = 0
            f2= 0
            f3=0
            f4 =0
            if(x == 0):
                f1 = w1
            else:
                f1 = w1/(x + 1e-8)

            if(y == 0):
                f2 = w2
            else:
                f2 = w2/(y + 1e-8)

            if(p == 0):
                f3 = w3
            else:
                f3 = w3/(p + 1e-8)
            
            if(q == 0):
                f4 = w4
            else:
                f4 = w4/(q + 1e-8)


            fitness = f1 + f2 +f3 + f4
            
            # fitness = round(fitness)
            population_fitness.append(fitness)
            
        return population_fitness

    def _mutate(self, individual, highest_fitness, avg_fitness):
        """ Randomly change the individual's characters with probability
        self.mutation_rate """
        n = np.random.randint(1,len(individual))
        random_row = []
        while n > 0:
            row = np.random.randint(0,len(individual))
            if row in random_row:
                continue
            else:
                random_row.append(row)
                n-=1
        
        while True:
            random_time_slot1 = np.random.randint(0,45)
            random_time_slot2 = np.random.randint(0,45)
            if random_time_slot1 != random_time_slot2:
                break
        
        for row in random_row:
            tmp_activity = individual[row][random_time_slot1]
            individual[row][random_time_slot1] = individual[row][random_time_slot2]
            individual[row][random_time_slot2] = tmp_activity
        
        return individual

    def transferProcess(self, parent1,parent2):
        timeslot = len(parent1[0])
        ruangan = len(parent1)
        all_ruangan = []
        array_of_timeslot = []
        for i in range(len(all_sesi)):
            array_of_timeslot.append(i)
        for i in range(ruangan):
            all_ruangan.append(i)

        n_parent1 = int(len(parent1[0]) * 80 / 100)
        n_parent2 = int(len(parent2[0]) * 20 / 100)
        # n_parent1 = 36
        # n_parent2 = 9
        

        all_indices = np.array(array_of_timeslot)
       
        random_t1_slot = np.random.choice(all_indices, size=n_parent1, replace=False)
        random_t2_slot = np.random.choice(all_indices, size=n_parent2, replace=False)
        random_t2_slot = np.setdiff1d(all_indices, random_t1_slot)[:n_parent2]

        child1 = [['' for j in range(timeslot)] for i in range(ruangan)]
        for time in random_t1_slot:
            if all_sesi[time] in self.unwanted_sesi:
                continue
            for room in range(ruangan):
                activity = parent1[room][time]
               
                if activity != '':
                    child1[room][time] = activity
                    self.transferred[activity] = True

        for time in random_t2_slot:
            if all_sesi[time] in self.unwanted_sesi:
                continue
            for room in range(ruangan):
                activity = parent2[room][time]  
                if activity != '' and self.transferred[activity] == False:
                    child1[room][time] = activity
                    self.transferred[activity] = True
        

        # APPLY HA FOR COURSE THAT HAS NOT BEEN TRANSFERRED
        len_unassigned = 0
        for key in self.transferred:
            if self.transferred[key] == False:
                len_unassigned = len_unassigned + 1
        
        for key in self.transferred:
            if self.transferred[key] == False:
                while True:
                    random_HA_time = np.random.choice(all_indices,size=1, replace=False)
                    random_HA_room = np.random.choice(np.array(all_ruangan), size=1, replace=False)
                    if child1[random_HA_room[0]][random_HA_time[0]] == '' and all_sesi[int(random_HA_time)] not in self.unwanted_sesi:
                        self.transferred[key] = True
                        child1[random_HA_room[0]][random_HA_time[0]] = key
                        break
        
        self.initTransferredActivity(True)
        return child1

    def _crossover(self, parent1, parent2):
        """ Create children from parents by crossover """
        child1 = self.transferProcess(parent1, parent2)
        child2 = self.transferProcess(parent2, parent1)
        return child1,child2
                

    
    def terminate(self, population_fitness):
        avg = sum(population_fitness) / len(population_fitness)
        opt_f = max(population_fitness)
        # 2nd Termination
        # if opt_f/avg > 0.9:
        #     return True
        # 3rd Termination
        if opt_f == min (population_fitness):
            return True
        return False

    def repairFunction(self, individu,x,y,z,p,q):
        timeslot = len(individu[0])
        ruangan = len(individu)
        duplicate_coordinates = []
        # while x > 0:
        count_perkuliahan_table = 0
        # print(timeslot, ruangan)
        # First constraint
        while x>0:
            # Tracking duplicates
            for time in range(timeslot):
                count_same_dosen = {}
                for room in range (ruangan):
                    activity = individu[room][time]
                    if(activity != ''):
                        dosen = activity[0:2]
                        count_same_dosen[dosen] =  count_same_dosen.get(dosen, 0) + 1
                        count_perkuliahan_table = count_perkuliahan_table + 1
                duplicate_in_timeslot = sum(count-1 for count in count_same_dosen.values() if count > 1 )
                x = x + duplicate_in_timeslot
                
                for room, activity in enumerate(individu):
                    if activity[time] != '':
                        dosen = activity[time][0:2]
                        if count_same_dosen[dosen] > 1:
                            duplicate_coordinates.append((room, time))
                            count_same_dosen[dosen] = 0
            
            #Repairing
            for item in duplicate_coordinates:
                first_value, second_value = item
                random_col = 0
                random_row = 0
                while True:
                    random_col = np.random.randint(0,len(individu[0]))
                    random_row = np.random.randint(0, len(individu))
                    if individu[random_row][random_col] == '':
                        individu[random_row][random_col] = individu[first_value][second_value]
                        x-=1
                        break
                
        return individu,x,y,z,p,q

    def run(self):
        # Initialize new population
        # Initialize new population
        self._initialize()
        # print("population\n",self.population)
        # p1 = 0.5
        # p2 = 0.5
        maximum_fitness = 0
        most_fit = [[]]
        iterations = 300
        for epoch in range(iterations):
            population_fitness = self._calculate_fitness()
            print(population_fitness)
            
        #     # print(x,y,z,p)
        #     # This is the indivdual
            fittest_individual = self.population[np.argmax(population_fitness)]

        #     # While this is the number
            highest_fitness = max(population_fitness)
            lowest_fitness = min(population_fitness)
            avg_fitness = (sum(population_fitness) / len(population_fitness))
            # print(avg_fitness)
        # #     # If we have found individual which matches the target => Done
            if highest_fitness >= maximum_fitness:
                maximum_fitness = highest_fitness
                most_fit = fittest_individual
        #     if epoch == iterations:
        #         break

        # #     # Set the probability that the individual should be selected as a parent
        # #     # proportionate to the individual's fitness.
            parent_probabilities = []
            # print("Fitness\n", population_fitness)
            # print("Highest:\n", highest_fitness)
            # print("Average:\n", avg_fitness)
            # print("Lowest:\n", lowest_fitness)
            for fitness in population_fitness:
                probability = 0
                if fitness >= avg_fitness:
                    probability = ((fitness-avg_fitness)/(highest_fitness-avg_fitness+1e-8)) 
                else:
                    probability =  ((avg_fitness-fitness)/(avg_fitness-lowest_fitness+1e-8))
                
                probability = probability / sum(population_fitness)
                # probability = probability * 100
                parent_probabilities.append(probability)
            # print(sum(parent_probabilities))
            # print(parent_probabilities)
        # #     # Determine the next generation
            new_population = []
            for i in np.arange(0, self.population_size):
        # #         # Select two parents randomly according to probabilities
                
                parent1_f, parent2_f = random.choices(population_fitness, k=2, weights=parent_probabilities)
                parent1_index = population_fitness.index(parent1_f)
                parent1 = self.population[parent1_index]
        #         # print(parent1)
                

                parent2_index = population_fitness.index(parent2_f)
                parent2 = self.population[parent2_index]

        # #         # Perform crossover to produce offspring
                child1, child2 = self._crossover(parent1, parent2)
        # #         # Save mutated offspring for next generation
                # print("Ini child hasil mutate")
                # print(self._mutate(child1,highest_fitness,avg_fitness))
                # print(self._mutate(child2,highest_fitness,avg_fitness))
                # new_population += [child1, child2]
                new_population += [self._mutate(child1,highest_fitness,avg_fitness), self._mutate(child2,highest_fitness,avg_fitness)]

            print ("[%d Epoch, Fitness: %.2f]" % (epoch,highest_fitness))
           
            self.population = new_population
            # print(self.population)

        if highest_fitness <= maximum_fitness:
            fittest_individual = most_fit
            highest_fitness = maximum_fitness
        print ("[%d Answer: '%s']\n [Fitness: %.2f]" % (epoch, fittest_individual, highest_fitness))
        # print("SKPB ", self.list_skpb)
        x,y,z,p,q = self._individuConstrain(fittest_individual)
        print("Before repaired: " ,x,y,z,p,q)
        repaired_individu,x,y,z,p,q = self.repairFunction(fittest_individual,x,y,z,p,q)
      
        # print(x,y,z,p,q)
        return self.parseJsn(repaired_individu,x,y,z,p,q,highest_fitness)

    def parseJsn(self, individual,x,y,z,p,q,highest_fitness):
        result = {
            "data": [],
            "skpb": [],
            "violated_constraint":{},
            "list_ruangan": [],
        }
      
        for time in range(len(individual[0])):
            sesi = all_sesi[time]
            for room in range(len(individual)):
                if individual[room][time] != "":
                    dosen = individual[room][time][0:2]
                    ruangan = self.data_ruangan[room]
                    
                    res = {
                        "dosen":dosen,
                        "mata_kuliah": individual[room][time][2:8],
                        "kelas": individual[room][time][8],
                        "ruangan": ruangan, #a
                        "sesi":sesi, #a
                        "time": time,
                        # "preferensi": dosenPrefensiDict[dosen],
                        "tipe": "jurusan",
                        # "rmk": self.data["data"][index]["rmk"]
                    }
                    result["data"].append(res)
        for skpb in self.list_skpb:
            res = {
                "kode_dosen": skpb[0:2],
                "kode_mk": skpb[2:8],
                "kelas": skpb[8],
                "ruangan": skpb[9:12],
                "sesi": skpb[12:15],
                "preferensi": [],
                "tipe": "SK",
                "rumpun": "SKPB"
            }
            result["skpb"].append(res)
        result["violated_constraint"]["first_constraint"] = x
        result["violated_constraint"]["second_constraint"] = y
        result["violated_constraint"]["third_constraint"] = z
        result["violated_constraint"]["fourth_constraint"] = p
        result["violated_constraint"]["fifth_constraint"] = q
        result["fitness"] = highest_fitness
        result['list_ruangan'] = self.data_ruangan
        result['unwanted_sesi'] = self.unwanted_sesi
        return result


