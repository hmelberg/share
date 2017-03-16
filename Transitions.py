import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('seaborn-notebook')
plt.style.use('ggplot')
#plt.style.use('fivethirtyeight')
import seaborn as sns
import math
import gc
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import random

from Inputs import *

# Transition probability sets
## From a0
# Transition probabilities
def tp_a0b0(age):
    return rb(age)
def tp_a0c0(age):
    return ro(age)
def tp_a0d0():
    return mo()
def tp_a0d1():
    return ma()
def tp_a0d2():
    return oo()
def tp_a0f0(age):
    return mb(age)

# Complete probability set from state
def from_a0(age):
    """
    Needs to be treated as array of occuring events
    """
    true_pr_a0 = [
        (1 - (tp_a0b0(age) + tp_a0c0(age) + tp_a0d0() + tp_a0d1() + tp_a0d2() + tp_a0f0(age))),
        tp_a0b0(age),
        tp_a0c0(age),
        tp_a0d0(),
        tp_a0d1(),
        tp_a0d2(),
        tp_a0f0(age)
                  ]
# Event for agent each step| transition probabilities, own age
    a0 = np.random.multinomial(1, pvals = true_pr_a0)
    return a0

## From a1
# Transition probabilties
def tp_a1c0(age):
    return ro(age)
def tp_a1d2():
    return oo()
def tp_a1f0(age):
    return mb(age)

# Complete probability set from state
def from_a1(age):
    """
    Needs to be treated as array of occuring events
    """
    true_pr_a1 = [
        (1 - (tp_a1c0(age) + tp_a1d2() + tp_a1f0(age))),
        tp_a1c0(age),
        tp_a1d2(),
        tp_a1f0(age)
                ]
# Event for agent each step | transition probabilities, own age
    a1 = np.random.multinomial(1, pvals = true_pr_a1)
    return a1

## From a2 

# Transition probabilities
def tp_a2b0r(age):
    return bcr(age)
def tp_a2b0(age):
    return rb(age)
def tp_a2d1():
    return ma()
def tp_a2f0(age):
    return mb(age)

# Complete probability set from state
def from_a2r(age):
    true_pr_a2r = [
        (1 - (tp_a2b0r(age) + tp_a2d1() + tp_a2f0(age))),
        tp_a2b0r(age),
        tp_a2d1(),
        tp_a2f0(age)
                ]
# Event for agent each step | transition probabilities, own age
    a2r = np.random.multinomial(1, pvals = true_pr_a2r)
    return a2r


# Complete probability set from state
def from_a2(age):
    """
    Needs to be treated as array of occuring events
    """
    true_pr_a2 = [
        (1 - (tp_a2b0(age) + tp_a2d1() + tp_a2f0(age))),
        tp_a2b0(age),
        tp_a2d1(),
        tp_a2f0(age)
                ]
# Event for agent each step | transition probabilities, own age
    a2 = np.random.multinomial(1, pvals = true_pr_a2)
    return a2

## From b0
# Transition probabilities
def tp_b0b1(age):
    return bc1(age)
def tp_b0b2(age):
    return bc2(age)
def tp_b0b3(age):
    return bc3(age)
def tp_b0b4(age):
    return bc4(age)

# Complete probability set from state
def from_b0(age):
    """
    Needs to be treated as array of occuring events, e.g. (from_b0[0]==1) means 
    agent transfers to stage 1 bc.
    """
    true_pr_b0 = [
        tp_b0b1(age),
        tp_b0b2(age),
        tp_b0b3(age),
        tp_b0b4(age)
    ]
# Event for agent each step | transition probabilities, own age
    b0 = np.random.multinomial(1, pvals = true_pr_b0)
    return b0

## From b1
# Transition probabilties
def tp_b1(age):
    return m_bc1(age)

def from_b1(age, time_in_b1):
    true_pr_b1 = [(1 - tp_b1(age)), tp_b1(age), 0]
    if time_in_b1>=10:
        true_pr_b1 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age        
    b1 = np.random.multinomial(1, pvals = true_pr_b1)
    return b1

## From b2
# Transition probabilites
def tp_b2(age):
    return m_bc2(age)

def from_b2(age, time_in_b2):
    """
    If returning 1: agent survives untill next cycle
    Else: agent transfers to state f1 (bc specific mortality)
    If agent stays for a period of t = 10 cycles, she transfer out to state_a1
    """
    true_pr_b2 = [(1 - tp_b2(age)),
                  tp_b2(age),
                  0
                  ]
    if time_in_b2>=10:
        true_pr_b2 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age        
    b2 = np.random.multinomial(1, pvals= true_pr_b2)
    return b2

## From b3
# Transition probabilites
def tp_b3(age):
    return m_bc3(age)

def from_b3(age, time_in_b3):
    """
    If returning 1: agent survives untill next cycle
    Else: agent transfers to state f1 (bc specific mortality)
    If agent stays for a period of t = 10 cycles, she transfer out to state_a1
    """
    true_pr_b3 = [(1 - tp_b3(age)),
                  tp_b3(age),
                  0
                  ]
    if time_in_b3>=10:
        true_pr_b3 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age
    b3 = np.random.multinomial(1, pvals= true_pr_b3)
    return b3

## From b4
# Transition probabilites
def tp_b4(age):
    return m_bc4(age)

def from_b4(age, time_in_b4):
    """
    If returning 1: agent survives untill next cycle
    Else: agent transfers to state f1 (bc specific mortality)
    If agent stays for a period of t = 10 cycles, she transfer out to state_a1
    """
    true_pr_b4 = [(1 - tp_b4(age)),
                  tp_b4(age),
                  0
                  ]
    if time_in_b4>=10:
        true_pr_b4 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age        
    b4 = np.random.multinomial(1, pvals= true_pr_b4)
    return b4

## From c0
# Transition probabilities
def tp_c0c1():
    return ocl()
def tp_c0c2():
    return ocr()
def tp_c0c3():
    return ocd()

def from_c0():
    """
    Returns array with 1 or 0 for transition to each stage of ovarian cancer from diagnosis
    """
    true_pr_c0 = [
        tp_c0c1(),
        tp_c0c2(),
        tp_c0c3()
    ]
# Event for agent each step | transition probabilities, own age    
    c0 = np.random.multinomial(1, pvals = true_pr_c0)
    return c0

## From c1
# Transition probabilities
def tp_c1f2(age):
    return m_ocl(age)

def from_c1(age, time_in_c1):
    """
    Returns 1 if agent survives untill next cycle, 0 means agent dies from ovarian cancer
    and transfer to state_f2
    If agent stays for a period of t = 10 cycles, she transfer out to state_a2
    """
    true_pr_c1 = [(1 - tp_c1f2(age)),
                  tp_c1f2(age),
                  0
                  ]
    if time_in_c1>=10: 
        true_pr_c1 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age        
    c1 = np.random.multinomial(1, pvals = true_pr_c1)
    return c1

## From c2
# Transition probabilities
def tp_c2f2(age):
    return m_ocr(age)

def from_c2(age, time_in_c2):
    """
    Returns 1 if agent survives untill next cycle, 0 means agent dies from ovarian cancer
    and transfer to state_f2
    If agent stays for a period of t = 10 cycles, she transfer out to state_a2
    """
    true_pr_c2 = [(1 - tp_c2f2(age)),
                  tp_c2f2(age),
                  0
                  ]
    if time_in_c2>=10: 
        true_pr_c2 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age        
    c2 = np.random.multinomial(1, pvals = true_pr_c2)
    return c2

## From c3
# Transition probabilities
def tp_c3f2(age):
    return m_ocd(age)

def from_c3(age, time_in_c3):
    """
    Returns 1 if agent survives untill next cycle, 0 means agent dies from ovarian cancer
    and transfer to state_f2
    If agent stays for a period of t = 10 cycles, she transfer out to state_a2
    """
    true_pr_c3 = [(1 - tp_c3f2(age)),
                  tp_c3f2(age),
                  0
                  ]
    if time_in_c3>=10: 
        true_pr_c3 = [0, 0, 1]
# Event for agent each step | transition probabilities, own age        
    c3 = np.random.multinomial(1, pvals = true_pr_c3)
    return c3

## From d0
# Transition probabilities
def tp_d0f0(age):
    return mb(age)

def from_d0(age):
    """
    Returns 1 if agent survives both masectomy and oophorectomy, 0 means agent dies from background mortality
    __Possible extension: include perioperative mortality for these surgeries
    """
    true_pr_d0 = (1 - tp_d0f0(age))
    d0 = np.random.binomial(1, p = true_pr_d0)
    return d0

## From d1
# Transition probabilties
def tp_d1f0(age):
    return mb(age)

def from_d1(age):
    """
    Returns 1 if agent survives prophylactic masectomy, 0 mean agent dies from background mortality
    __Possible extension: include perioperative mortality for pro. masec
    """
    true_pr_d1 = (1 - tp_d1f0(age))
# Event for agent each step | transition probabilities, own age    
    d1 = np.random.binomial(1, p = true_pr_d1)
    return d1

## From d2
# Transition probabilties
def tp_d2f0(age):
    return mb(age)

def from_d2(age):
    """
    Returns 1 if agent survives prophylactic oophorectomy, 0 mean agent dies from background mortality
    __Possible extension: include perioperative mortality for pro. oophore
    """
    true_pr_d2 = (1 - tp_d2f0(age))
# Event for agent each step | transition probabilities, own age    
    d2 = np.random.binomial(1, p = true_pr_d2)
    return d2

# From e0
def tp_e0f0(age):
    return mb(age)

def from_e0(age):
    """
    Returns 1 if agent survives untill next cycle, 0 mean agent dies from background mortaltiy
    State accessible only via d0, effectively removing increased mortaltiy risk from cancer in these organs
    """    
    true_pr_e0 = (1 - tp_e0f0(age))
# Event for agent each step | transition probabilities, own age    
    e0 = np.random(1, p = true_pr_e0)
    return e0
