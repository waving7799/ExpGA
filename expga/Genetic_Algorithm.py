import numpy as np
import pandas as pd


#classifier_name = 'Random_Forest_standard_unfair.pkl'
#model = joblib.load(classifier_name)

class GA():
    # input:
    #     nums: m * n  n is nums_of x, y, z, ...,and m is population's quantity
    #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
    #     DNA_SIZE is binary bit size, None is auto
    def __init__(self, nums, bound, func,  DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        nums= np.array(nums)
        bound = np.array(bound)
        self.bound = bound
        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        bits = np.ceil(np.log2(var_len + 1))

        if DNA_SIZE == None:
            DNA_SIZE = int(np.max(bits))
        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)
        # POP = np.zeros((*nums.shape, DNA_SIZE))
        # for i in range(nums.shape[0]):
        #     for j in range(nums.shape[1]):
        #         num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
        #         POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        # self.POP = POP
        self.POP = nums

        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        # self.importance = imp

    # def translateDNA(self):
    #     W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
    #     binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
    #     for i in range(binary_vector.shape[0]):
    #         for j in range(binary_vector.shape[1]):
    #             binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
    #             binary_vector[i, j] += self.bound[j][0]
    #     return binary_vector
    def get_fitness(self, non_negative=False):
        # result = self.func(*np.array(list(zip(*self.translateDNA()))))
        result = [self.func(self.POP[i]) for i in range(len(self.POP))]
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        return result

    def select(self):
        fitness = self.get_fitness()
        fit = [item[0] for item in fitness]
        # print(fit)
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fit / np.sum(fit))]
        pop_str = []
        for pop in self.POP:
            temp = []
            for x in pop:
                temp.append(str(x))
            pop_str.append(temp)
        pop_str = ["".join(x) for x in pop_str]
        # print(len(set(pop_str)))

    def crossover(self):
        k=0
        for people in self.POP:
            # imp = [abs(x) for x in self.importance[k]]
            # k += 1
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, len(self.bound))
                end_points = np.random.randint(0, len(self.bound)-cross_points)
                people[cross_points:end_points] = self.POP[i_, cross_points:end_points]

            # if np.random.rand() < self.cross_rate:
            #     i_ = np.random.randint(0, self.POP.shape[0], size=1)
            #     n = np.random.randint(0, len(people), size=1)
            #     # n=1
            #     cross_points = np.random.choice(np.arange(len(self.bound)), size=n, replace=False,
            #                                  p=imp / np.sum(imp))
            #     for j in cross_points:
            #         people[j] = self.POP[i_, j]

    def mutate(self):
        for people in self.POP:
            for point in range(self.DNA_SIZE):
                if np.random.rand() < self.mutation:
                    # var[point] = 1 if var[point] == 0 else 1
                    people[point] = np.random.randint(self.bound[point][0],self.bound[point][1])

    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

    def reset(self):
        self.POP = self.copy_POP.copy()

    def log(self):
        # return pd.DataFrame(np.hstack((self.POP, self.get_fitness())),
        #                     columns=['x{i}' for i in range(len(self.bound))] + ['F'])
        pop_str = []
        for pop in self.POP:
            temp = []
            for x in pop:
                temp.append(str(x))
            pop_str.append(temp)
        return pop_str, self.get_fitness()



# if __name__ == '__main__':
#     nums = [[3,0,10,3,1,6,3,0,1,0,0,40,0],[4,3,20,13,2,5,3,0,0,0,0,50,0],[3,0,14,1,0,4,2,4,1,0,0,80,0],[5,0,5,3,1,0,5,0,0,0,0,40,0]]
#     bound = config.input_bounds
#     # func = lambda x, y: x*np.cos(2*np.pi*y)+y*np.sin(2*np.pi*x)
#     DNA_SIZE = len(bound)
#     cross_rate = 0.7
#     mutation = 0.01
#     ga = GA(nums=nums, bound=bound, func=evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate, mutation=mutation)
#     res = ga.log()
#     print(res)
#     for i in range(10):
#         ga.evolution()
#         res = ga.log()
#         print(res)