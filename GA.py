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
    for j in range(1, 10):
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
        # f = open('test.json')
        self.data = raw_data
        self.unwanted_sesi = self.data["unwanted_sesi"]
        self.data_ruangan = self.data["ruangan"]
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
        # print(preferensiObj)
        for hari in preferensiObj["hari"]:
            tempSesi = tempSesi + hariDict[hari]
            for sesi in preferensiObj["sesi"]:
                listPrefrensi.append(tempSesi + sesiDict[sesi])
            tempSesi = ""

        return listPrefrensi

    def _initIndividu(self):
        list_perkuliahan = []
        banyak_perkuliahan = len(self.data['data'])
        random_sesi = []
        random_ruangan = []

        # Mengambil sesi/ruangan random dari data sesuai dengan
        # kebutuhan perkuliahan, jika banyak perkuliahan > sesi/ruangan
        # maka ambil semua data sesi/ruangan, jika tidak maka
        # cukup ambil sebanyak perkuliahan yang ada
        if (banyak_perkuliahan > len(self.data_ruangan)):
            # random_ruangan = random.sample(data_ruangan, len(data_ruangan))
            random_ruangan = np.random.choice(a=self.data_ruangan, size=len(self.data_ruangan), replace=False)
        else:
            random_ruangan = np.random.choice(a=self.data_ruangan, size=len(banyak_perkuliahan) + 1, replace=False)
            # random_ruangan = random.sample(data_ruangan, banyak_perkuliahan)

        if (banyak_perkuliahan > len(self.sesi)):
            random_sesi = np.random.choice(a=self.sesi, size=len(self.sesi), replace=False)
        else:
            random_sesi = np.random.choice(a=self.sesi, size=len(banyak_perkuliahan) + 1, replace=False)

        # Iteration for sesi and ruangan
        j = 0  # ruangan
        k = 0  # sesi

        # Inisiasi untuk SKPB
        self.list_skpb = []
        for data in self.data['data']:
            individuSesi = self.preferensiToSesi(data["preferensi"])
            if data['dosen'] not in dosenPrefensiDict:
                dosenPrefensiDict[data['dosen']] = individuSesi
            if data['mata_kuliah'][0:2] != "UG":
                list_perkuliahan.append(
                    data['dosen'] + data['mata_kuliah'] + data['kelas'] + random_ruangan[j] + random_sesi[k])
                j = j + 1
                k = k + 1
                if j >= len(random_ruangan):
                    j = 0
                if k >= len(random_sesi):
                    k = 0

            else:  # Input data SKPB
                self.list_skpb.append(
                    data['dosen'] + data['mata_kuliah'] + data['kelas'] + data['ruangan'] + data['sesi'])

        # list_perkuliahan sudah menjadi Individu
        return list_perkuliahan

    def removeUnwantedSesi(self):
        for unwanted in self.unwanted_sesi:
            if unwanted in default_sesi:
                default_sesi.remove(unwanted)
            if unwanted in all_sesi:
                all_sesi.remove(unwanted)
        return default_sesi

    def _initialize(self):
        """ Initialize population with random strings """
        self.population = []
        self.sesi = self.removeUnwantedSesi()

        for _ in range(self.population_size):
            # Enter individu to population
            individual = self._initIndividu()
            self.population.append(individual)

    # Return number of violation [x,y,z] for an individual
    def _individuConstrain(self, individu):
        x = 0  # First to third constraint
        y = 0
        z = 0
        p = 0  # fourth constraint
        q = 0  # prefrensi
        for i in range(len(individu)):
            cnt_day = 0
            for j in range(i + 1, len(individu)):
                if individu[i][2:9] != individu[j][2:9]:
                    # First constraint: Dosen yang sama tidak bisa mengajar 2 MK di waktu yang sama
                    if individu[i][0:2] == individu[j][0:2]:  # Sama dosen
                        if abs(int(individu[i][12:15]) - int(individu[j][12:15])) < 2:  # Sama hari sesi
                            x = x + 1
                    # Second constraint: Ruangan yang sama tidak dapat dipakai di waktu yang sama
                    if individu[i][9:12] == individu[j][9:12]:  # Sama ruangan
                        if abs(int(individu[i][12:15]) - int(individu[j][12:15])) < 2:  # Sama hari sesi
                            y = y + 1

                    # Third constraint: Satu dosen maksimal mengajar 2x sehari
                    if individu[i][0:2] == individu[j][0:2]:  # Sama dosen
                        if individu[i][12] == individu[j][12]:  # Sama hari
                            cnt_day = cnt_day + 1
                    if cnt_day > 2:
                        z = z + cnt_day - 2

            p = p + self.skpbConstraint(individu[i])
            q = q + self.prefrensiConstraint(individu[i])

        return [x, y, z, p, q]

    def prefrensiConstraint(self, gen):
        qnow = 0
        if gen[12:15] not in dosenPrefensiDict[gen[0:2]]:
            qnow = qnow + 1
        return qnow

    def skpbConstraint(self, gen):
        pnow = 0
        for i in range(len(self.list_skpb)):
            # Fourth Constraint: Sesi SKPB dan jurusan minimal 1 jam
            if gen[2:4] == "SK":
                if abs(int(gen[12:15]) - int(self.list_skpb[i][12:15])) < 2:
                    pnow = pnow + 1
        return pnow

    def individuFitness(self, individual):
        w1 = 34
        w2 = 34
        w3 = 16
        w4 = 15
        w5 = 1
        x, y, z, p, q = self._individuConstrain(individual)
        fitness = w1 / (x + 1) + w2 / (y + 1) + w3 / (z + 1) + w4 / (p + 1) + w5 / (q + 1)
        return [fitness, x, y, z, p, q]

    def _calculate_fitness(self):
        """ Calculates the fitness of each individual in the population """
        population_fitness = []
        w1 = 30
        w2 = 30
        w3 = 20
        w4 = 19
        w5 = 1
        for individual in self.population:
            # loss: Array[x,y,z,p,q]
            x, y, z, p, q = self._individuConstrain(individual)
            # print("x,y,z,p,q", x,y,z,p,q)

            fitness = w1 / (x + 1) + w2 / (y + 1) + w3 / (z + 1) + w4 / (p + 1) + w5 / (q + 1)

            # fitness = round(fitness)
            population_fitness.append(fitness)

        return population_fitness

    def _mutate(self, individual, highest_fitness, avg_fitness):
        """ Randomly change the individual's characters with probability
        self.mutation_rate """
        p3 = random.random()
        p4 = random.random()
        individual = list(individual)
        fitness, xm, ym, zm, ppm, qm = self.individuFitness(individual)
        pm = 0
        if fitness >= avg_fitness:
            pm = p3 * (highest_fitness - fitness) / (highest_fitness - avg_fitness + 1e8) + 1e8
        else:
            pm = p4
        for j in range(len(individual)):
            # Make change with probability mutation_rate
            if np.random.random() < pm:
                if xm > ym or qm > 0:
                    temp_str = individual[j][0:12]
                    temp_str = temp_str + random.sample(all_sesi, 1)[0]
                    individual[j] = temp_str
                else:
                    temp_str = individual[j][0:9]
                    temp_str = temp_str + random.sample(self.data_ruangan, 1)[0] + individual[j][12:15]
                    individual[j] = temp_str
                # else:
                #     temp_str = individual[j][0:9]
                #     temp_str = temp_str + random.sample(data_ruangan,1)[0] + random.sample(sesi,1)[0]
                #     individual[j] = temp_str
        # Return mutated individual as string
        return individual

    def _crossover(self, parent1, parent2):
        """ Create children from parents by crossover """
        # Select random crossover point
        cross_i = np.random.randint(0, len(parent1))
        child1 = parent1[:cross_i] + parent2[cross_i:]
        child2 = parent2[:cross_i] + parent1[cross_i:]
        return child1, child2

    def terminate(self, population_fitness):
        avg = sum(population_fitness) / len(population_fitness)
        opt_f = max(population_fitness)
        # 2nd Termination
        # if opt_f/avg > 0.9:
        #     return True
        # 3rd Termination
        if opt_f == min(population_fitness):
            return True
        return False

    def run(self):
        # Initialize new population
        self._initialize()
        iterations = 100
        p1 = 0.5
        p2 = 0.5
        maximum_fitness = 0
        most_fit = []
        iterations = 100
        for epoch in range(iterations):
            population_fitness = self._calculate_fitness()
            # print (population_fitness)

            # print(x,y,z,p)
            # This is the indivdual
            fittest_individual = self.population[np.argmax(population_fitness)]
            # While this is the number
            highest_fitness = max(population_fitness)
            lowest_fitness = min(population_fitness)
            avg_fitness = round(sum(population_fitness) / len(population_fitness))
            #     # If we have found individual which matches the target => Done
            if highest_fitness >= maximum_fitness:
                maximum_fitness = highest_fitness
                most_fit = fittest_individual
            if epoch == iterations:
                break

            #     # Set the probability that the individual should be selected as a parent
            #     # proportionate to the individual's fitness.
            parent_probabilities = []
            # print("Fitness\n", population_fitness)
            # print("Highest:\n", highest_fitness)
            # print("Average:\n", avg_fitness)
            # print("Lowest:\n", lowest_fitness)
            for fitness in population_fitness:
                probability = 0
                if fitness >= avg_fitness:
                    probability = ((fitness - avg_fitness) / (highest_fitness - avg_fitness + 1e8)) + 1e8
                else:
                    probability = ((avg_fitness - fitness) / (avg_fitness - lowest_fitness + 1e8)) + 1e8

                # probability = round(probability, 3)
                probability = probability / sum(population_fitness)
                # probability = probability * 100
                parent_probabilities.append(probability)
            # print(sum(parent_probabilities))
            # print(parent_probabilities)
            #     # Determine the next generation
            new_population = []
            for i in np.arange(0, self.population_size, 2):
                #         # Select two parents randomly according to probabilities

                parent1_f, parent2_f = random.choices(population_fitness, k=2, weights=parent_probabilities)
                parent1_index = population_fitness.index(parent1_f)
                parent1 = self.population[parent1_index]
                # print(parent1)

                parent2_index = population_fitness.index(parent2_f)
                parent2 = self.population[parent2_index]

                #         # Perform crossover to produce offspring
                child1, child2 = self._crossover(parent1, parent2)
                #         # Save mutated offspring for next generation
                new_population += [self._mutate(child1, highest_fitness, avg_fitness),
                                   self._mutate(child2, highest_fitness, avg_fitness)]

            # print ("[%d Epoch, Fitness: %.2f]" % (epoch,highest_fitness))

            self.population = new_population
            # print(self.population)

        if highest_fitness <= maximum_fitness:
            fittest_individual = most_fit
            highest_fitness = maximum_fitness
        # print ("[%d Answer: '%s']\n [Fitness: %.2f]" % (epoch, fittest_individual, highest_fitness))
        # print("SKPB ", self.list_skpb)
        x, y, z, p, q = self._individuConstrain(fittest_individual)
        # print(x,y,z,p,q)

        return self.parseJsn(fittest_individual,x,y,z,p,q,highest_fitness)

    def parseJsn(self, individual,x,y,z,p,q,highest_fitness):
        result = {
            "data": [],
            "skpb": [],
            "violated_constraint":{},
        }
        for index in range(len(individual)):
            res = {
                "dosen": individual[index][0:2],
                "mata_kuliah": individual[index][2:8],
                "kelas": individual[index][8],
                "ruangan": individual[index][9:12],
                "sesi": individual[index][12:15],
                "preferensi": self.data["data"][index]["preferensi"],
                "tipe": "jurusan",
                "rmk": self.data["data"][index]["rmk"]
            }
            result["data"].append(res)
        for skpb in self.list_skpb:
            res = {
                "dosen": skpb[0:2],
                "mata_kuliah": skpb[2:8],
                "kelas": skpb[8],
                "ruangan": skpb[9:12],
                "sesi": skpb[12:15],
                "preferensi": [],
                "tipe": "SK",
                "rmk": "SKPB"
            }
            result["skpb"].append(res)
        result["violated_constraint"]["first_constraint"] = x
        result["violated_constraint"]["second_constraint"] = y
        result["violated_constraint"]["third_constraint"] = z
        result["violated_constraint"]["fourth_constraint"] = p
        result["violated_constraint"]["fifth_constraint"] = q
        result["fitness"] = highest_fitness
        return result


