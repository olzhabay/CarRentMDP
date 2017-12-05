import numpy as np
import math
import time
import matplotlib.pyplot as plt
from itertools import product
import mdptoolbox.example, mdptoolbox
import plotly.plotly as py
import plotly.tools as tls
from matplotlib import cm
from plotly.graph_objs import *
from mpl_toolkits.mplot3d import Axes3D

# from https://plot.ly/matplotlib/trisurf/
def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    # x, y, z are lists of coordinates of the triangle vertices
    # simplices are the simplices that define the triangularization;
    # simplices  is a numpy array of shape (no_triangles, 3)
    # insert here the  type check for input data

    points3D = np.vstack((x, y, z)).T
    tri_vertices = map(lambda index: points3D[index], simplices)  # vertices of the surface triangles
    zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]  # mean values of z-coordinates of
    # triangle vertices
    min_zmean = np.min(zmean)
    max_zmean = np.max(zmean)
    facecolor = [map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]
    I, J, K = tri_indices(simplices)

    triangles = Mesh3d(x=x,
                       y=y,
                       z=z,
                       facecolor=facecolor,
                       i=I,
                       j=J,
                       k=K,
                       name=''
                       )

    if plot_edges is None:  # the triangle sides are not plotted
        return Data([triangles])
    else:
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        lists_coord = [[[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)]

        # define the lines to be plotted
        lines = Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          line=Line(color='rgb(50,50,50)', width=1.5)
                          )
        return Data([triangles, lines])


def poisson_prob(lmbda, x):
    return math.exp(-lmbda) * pow(lmbda, x) / math.factorial(x)


class CarRent:
    def __init__(self, lambda_return1, lambda_return2, lambda_rent1, lambda_rent2, max_cars,
                 gamma=0.9, request_reward=10, move_cost=2, theta=1 * pow(10, -4)):
        self.max_cars = max_cars
        self.lambda_return1 = lambda_return1
        self.lambda_return2 = lambda_return2
        self.lambda_rent1 = lambda_rent1
        self.lambda_rent2 = lambda_rent2
        self.gamma = gamma
        self.theta = theta
        self.finished = False
        self.rent_reward = request_reward
        self.move_cost = move_cost
        self.transitions = {}
        self.rewards = {}
        self.states = list(product(range(self.max_cars + 1), repeat=2))
        self.actions = dict(((n1, n2), list(range(max(-5, -n2), min(n1, 5) + 1))) for n1, n2 in self.states)
        self.probabilities = {}
        self.values = [dict(zip(self.states, [0] * len(self.states)))]
        self.policies = [dict(zip(self.states, [0] * len(self.states)))]
        for lmbda in [self.lambda_rent1, self.lambda_rent2, self.lambda_return1, self.lambda_return2]:
            self.probabilities[lmbda] = [poisson_prob(lmbda, x) for x in range(30)]

    def CalculateExpectation(self, num_cars, lambda_rent, lambda_return):
        rewards = 0
        left_probability = [0] * (self.max_cars + 1)
        for x, xprob in enumerate(self.probabilities[lambda_rent]):
            rent = min(x, num_cars)
            rewards += self.rent_reward * rent * xprob
            for y, yprob in enumerate(self.probabilities[lambda_return]):
                left = min(num_cars - rent + y, self.max_cars)
                left_probability[left] += xprob * yprob
        return rewards, left_probability

    def CalculateTRModels(self):
        for s in self.states:
            n1, n2 = s
            self.rewards[s] = {}
            self.transitions[s] = {}
            for a in self.actions[s]:
                reward1, prob_dist1 = self.CalculateExpectation(n1 - a, self.lambda_rent1, self.lambda_return1)
                reward2, prob_dist2 = self.CalculateExpectation(n2 + a, self.lambda_rent2, self.lambda_return2)
                self.rewards[s][a] = reward1 + reward2 - self.move_cost * abs(a)
                self.transitions[s][a] = {}
                for cars1, prob1 in enumerate(prob_dist1):
                    for cars2, prob2 in enumerate(prob_dist2):
                        self.transitions[s][a][cars1, cars2] = prob1 * prob2

    def Iterate(self):
        if self.finished: return
        self.PolicyEvaluation()
        self.GreedyPolicy()
        self.finished = all(self.policies[-1] == self.policies[-2] for s in self.states)

    def PolicyEvaluation(self):
        V = self.values[-1].copy()
        P = self.policies[-1].copy()
        while True:
            V_ = {}
            for state in self.states:
                action = P[state]
                V_[state] = self.rewards[state][action]
                V_[state] += sum(self.gamma * self.transitions[state][action][ns] * V[ns] for ns in self.states)
            delta = max(abs(V[state] - V_[state]) for state in self.states)
            V = V_
            if delta < self.theta:
                self.values.append(V_)
                return

    def GreedyPolicy(self):
        global best_action
        V = self.values[-1].copy()
        P = {}
        for state in self.states:
            best_reward = -10000
            for action in self.actions[state]:
                reward = self.rewards[state][action]
                reward += sum(
                    self.gamma * self.transitions[state][action][state_] * V[state_] for state_ in self.states)
                if reward > best_reward:
                    best_action = action
                    best_reward = reward
            P[state] = best_action
        self.policies.append(P)

    def PlotPolicy(self):
        length = self.max_cars + 1
        data = [[0] * length for x in range(length)]
        for x, y in self.states:
            data[x][y] = self.policies[-1][x, y]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Policy '+str(len(self.policies)-1))

        plotly_fig = tls.mpl_to_plotly( fig )
        trace = dict(z=data, type="heatmap", zmin=-5, zmax=5)
        plotly_fig['data'] = [trace]

        plotly_fig['layout']['xaxis'].update({'autorange':True})
        plotly_fig['layout']['yaxis'].update({'autorange':True})

        filename = 'mdp-policy#' + str(len(self.policies)-1)
        plot_url = py.plot(plotly_fig, filename=filename)

    def PlotValue(self, iter):
        x, y = zip(*self.states)
        z = [self.values[iter][x_,y_] for x_,y_ in zip(x,y)]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Value ' + str(iter))
        ax.set_ylabel('# of cars@loc1')
        ax.set_xlabel('# of cars@loc2')
        ax.scatter(x, y, z, zdir='z')
        plt.show()



jacks = CarRent(lambda_return1=3, lambda_return2=2,
                lambda_rent1=3, lambda_rent2=4, max_cars=20)
jacks.CalculateTRModels()
while not jacks.finished:
    #jacks.PlotPolicy()
    jacks.Iterate()
jacks.PlotValue(4)
