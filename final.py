#!/usr/bin/env python
# coding: utf-8

# # Read and clean data

# In[81]:


import csv
import os
import re
import random
from typing import List, Dict


# In[45]:


#data cleaning: change speaker's coding from gdx into x
pt = re.compile(r"\D+\d+")
a="gd23"
b="23"


# In[46]:


re.fullmatch(pt, a)


# In[47]:


re.fullmatch(pt, b)


# In[80]:


#def 
pt_spk = re.compile(r"\D+\d+")
pt_measure =re.compile(r"\D+")
measure_ls = []
keys_dic = []
with open("fricative_data.txt", "r") as source:
    with open("fricative_data_test.txt", "w") as sink:
        for row in csv.DictReader(source, delimiter = "\t"):
            if re.fullmatch(pt_spk, row["speaker"]):
                row["speaker"] = row["speaker"].lstrip("gd")
            if re.fullmatch(pt_measure, row["duration"]):
                row["duration"]= "missing"
                row["intensity"] = "missing"
                row["cog"] = "missing"
                row["sdev"] = "missing"
                row["skew"] = "missing"
                row["kurt"] = "missing"
            measure_ls.append(row)
            for key in row:
                if (key in keys_dic):
                    break
                else:
                    keys_dic.append(key)
        writer = csv.DictWriter(sink, fieldnames=keys_dic, delimiter = "\t")
        writer.writeheader()
        writer.writerows(measure_ls)


# In[87]:


ls = []
for i in range(30): 
    ls.append(random.randrange(1, 30, 1))
ls


# In[90]:


ten = len(ls) // 10
ls[:ten]
ls[ten:]

