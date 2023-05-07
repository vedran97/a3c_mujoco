from circular_trajectory import *
import numpy as np
import sympy as sp
_,_,_,_,_,_,T = getTransformationMatrixAndSymbols()

theta = [theta1, theta2, theta3, theta4, theta5, theta6]
T_func = sp.lambdify(theta, T, modules='numpy')


def calculateReward(targetThetas,currentAngles):
    xEE = T_func(*targetThetas)
    posEE =xEE[:3,3]
    posEE = np.array(posEE).astype(np.float64)

    currentEEF = T_func(*currentAngles)
    currentEEFPos = currentEEF[:3,3]
    currentEEFPos = np.array(currentEEFPos).astype(np.float64)

    reward = -1*np.linalg.norm(currentEEFPos-posEE)
    return reward