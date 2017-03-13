import numpy as np
import pandas as pd
import random
import math
import gc

from Evidence_synthesis import life_exp

m = pd.DataFrame(data = life_exp)

def mb(age):
    """
    Function returns probability of mortality in Norwegian pop. based on life-expectancy for entered age
    """
    p = life_exp.loc[(life_exp.Age==age),['p_die'] ]
    return p.p_die

from Evidence_synthesis import p_recurrence
rec = pd.DataFrame(data = p_recurrence)

def rb(age):
    """
    Funtction returns probability of breast cancer recurrence for index patient for entered age
    """
    if (age < 40): p = rec.recurrence[0]
    if (age > 39) & (age < 50): p = rec.recurrence[1]
    if (age > 49) & (age < 60): p = rec.recurrence[2]
    if (age > 59) & (age < 70): p = rec.recurrence[3]
    if (age >= 70): p = rec.recurrence[4]
    return p



from Evidence_synthesis import p_atrisk_to_bc
bc = pd.DataFrame(data = p_atrisk_to_bc)

def bcr(age):
    """
    Funtction returns probability of breast cancer incidence for relative for entered age
    """
    if (age > 14) & (age < 20): p = bc.prob_of_bc[0]
    if (age > 19) & (age < 25): p = bc.prob_of_bc[1]
    if (age > 24) & (age < 30): p = bc.prob_of_bc[2]
    if (age > 29) & (age < 35): p = bc.prob_of_bc[3]
    if (age > 34) & (age < 40): p = bc.prob_of_bc[4]
    if (age > 39) & (age < 45): p = bc.prob_of_bc[5]
    if (age > 44) & (age < 50): p = bc.prob_of_bc[6]
    if (age > 49) & (age < 55): p = bc.prob_of_bc[7]
    if (age > 54) & (age < 60): p = bc.prob_of_bc[8]
    if (age > 59) & (age < 65): p = bc.prob_of_bc[9]
    if (age > 64) & (age < 70): p = bc.prob_of_bc[10]
    if (age > 69) & (age < 75): p = bc.prob_of_bc[11]
    if (age > 74) & (age < 80): p = bc.prob_of_bc[12]
    if (age > 79) & (age < 85): p = bc.prob_of_bc[13]
    if (age > 84): p = bc.prob_of_bc[14]
    return p


from Evidence_synthesis import p_atrisk_to_oc 
roc = pd.DataFrame(data = p_atrisk_to_oc)

def ro(age):
    """
    Function returns probability of ovarian cancer for anyone for entered age
    """
    if (age < 40): pm = 0
    if (age > 39) & (age < 45): pm = roc.prob_of_oc[0]
    if (age > 44) & (age < 50): pm = roc.prob_of_oc[1]
    if (age > 49) & (age < 55): pm = roc.prob_of_oc[2]
    if (age > 54) & (age < 60): pm = roc.prob_of_oc[3]
    if (age > 59) & (age < 65): pm = roc.prob_of_oc[4]
    if (age > 64) & (age < 70): pm = roc.prob_of_oc[5]
    if (age > 69) & (age < 75): pm = roc.prob_of_oc[6]
    if (age > 74) & (age < 80): pm = roc.prob_of_oc[7]
    if (age > 79) & (age < 85): pm = roc.prob_of_oc[8]
    if (age > 84): pm = roc.prob_of_oc[9]
    return pm


from Evidence_synthesis import p_die_bc 
mr_bc = pd.DataFrame(data = p_die_bc)


def m_bc1(age):
    """
    Function returns probabiltiy of death with stage 1 breast cancer in Norway for entered age
    """
    p = mr_bc.loc[(mr_bc.Age==age),  ['p_die_stI']]
    return p.p_die_stI

def m_bc2(age):
    """
    Function returns probabiltiy of death with stage 2 breast cancer in Norway for entered age
    """
    p = mr_bc.loc[(mr_bc.Age==age),  ['p_die_stII']]
    return p.p_die_stII

def m_bc3(age):
    """
    Function returns probabiltiy of death with stage 3 breast cancer in Norway for entered age
    """
    p = mr_bc.loc[(mr_bc.Age==age),  ['p_die_stIII']]
    return p.p_die_stIII

def m_bc4(age):
    """
    Function returns probabiltiy of death with stage 4 breast cancer in Norway for entered age
    """
    p = mr_bc.loc[(mr_bc.Age==age),  ['p_die_stIV']]
    return p.p_die_stIV


from Evidence_synthesis import p_die_oc
mr_oc = pd.DataFrame(data = p_die_oc)

def m_ocl(age):
    """
    Function returns probabiltiy of death with local ovarian cancer in Norway for entered age
    """
    p = mr_oc.loc[(mr_oc.Age==age),  ['p_die_local_oc']]
    return p.p_die_local_oc

def m_ocr(age):
    """
    Function returns probabiltiy of death with regional ovarian cancer in Norway for entered age
    """
    p = mr_oc.loc[(mr_oc.Age==age),  ['p_die_regional_oc']]
    return p.p_die_regional_oc


def m_ocd(age):
    """
    Function returns probabiltiy of death with distant ovarian cancer in Norway for entered age
    """
    p = mr_oc.loc[(mr_oc.Age==age),  ['p_die_distant_oc']]
    return p.p_die_distant_oc


from Evidence_synthesis import p_stage_age
dist_bc = pd.DataFrame(data = p_stage_age)

def bc1(age):
    """
    Function returns probability of being in stage 1 breast cancer at point of _primary_ breast cancer, for entered age
    """
    if (age > 14) & (age < 50): p = dist_bc.stageI[0]
    if (age > 49) & (age < 70): p = dist_bc.stageI[1]
    if (age > 69): p = dist_bc.stageI[2]
    return p

def bc2(age):
    """
    Function returns probability of being in stage 2 breast cancer at point of _primary_ breast cancer, for entered age
    """
    if (age > 14) & (age < 50): p = dist_bc.stageII[0]
    if (age > 49) & (age < 70): p = dist_bc.stageII[1]
    if (age > 69): p = dist_bc.stageII[2]
    return p

def bc3(age):
    """
    Function returns probability of being in stage 3 breast cancer at point of _primary_ breast cancer, for entered age
    """
    if (age > 14) & (age < 50): p = dist_bc.stageIII[0]
    if (age > 49) & (age < 70): p = dist_bc.stageIII[1]
    if (age > 69): p = dist_bc.stageIII[2]
    return p

def bc4(age):
    """
    Function returns probability of being in stage 4 breast cancer at point of _primary_ breast cancer, for entered age
    """
    if (age > 14) & (age < 50): p = dist_bc.stageIV[0]
    if (age > 49) & (age < 70): p = dist_bc.stageIV[1]
    if (age > 69): p = dist_bc.stageIV[2]
    return p


from Evidence_synthesis import p_ocstage_at_diagn
dist_oc = pd.DataFrame(data = p_ocstage_at_diagn)

def ocl():
    """
    Function returns probability of being diagnosed with local disease | diagnosis of ovarian cancer
    """
    return dist_oc.dist[0]

def ocr():
    """
    Function returns probabiltiy of being diagnosed with regional disease | diagnosis of ovarian cancer
    """
    return dist_oc.dist[1]

def ocd():
    """
    Function returns probabiltiy of being diagnosed with distant disease | diagnosis of ovarian cancer
    """
    return dist_oc.dist[2]

def mo():
    """
    Funtcion returns probability of undergoing both masectomy and oophorectomy
    """
    return 0

def ma():
    """
    Function returns probabiltiy of undergoing only masectomy
    """
    return 0

def oo():
    """
    Function returns probability of undergoing only oophorectomy
    """
    return 0
