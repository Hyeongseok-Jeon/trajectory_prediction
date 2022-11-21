import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import csv

matplotlib.use('Qt5Agg')
tree_MIA = ET.parse('dataset/Argoverse/map_files/pruned_argoverse_MIA_10316_vector_map.xml')
tree_PIT = ET.parse('dataset/Argoverse/map_files/pruned_argoverse_PIT_10314_vector_map.xml')
tree_HMC = ET.parse('dataset/HMC/map_files/pruned_argoverse_HMC_10317_vector_map.xml')
root_MIA = tree_MIA.getroot()
root_PIT = tree_PIT.getroot()
root_HMC = tree_HMC.getroot()

map_point = []
for child in root_MIA:
    if child.tag == 'node':
        pt = [float(child.attrib['x']), float(child.attrib['y'])]
        map_point.append(pt)
for child in root_PIT:
    if child.tag == 'node':
        pt = [float(child.attrib['x']), float(child.attrib['y'])]
        map_point.append(pt)
for child in root_HMC:
    if child.tag == 'node':
        pt = [float(child.attrib['x']), float(child.attrib['y'])]
        map_point.append(pt)
map_point = np.array(map_point)

data_root_HMC = 'dataset/HMC'
data_list_HMC_1 = glob.glob(data_root_HMC+'/train/data/*.csv')
data_list_HMC_2 = glob.glob(data_root_HMC+'/test_obs/data/*.csv')
data_list_HMC_3 = glob.glob(data_root_HMC+'/val/data/*.csv')
data_list_HMC = data_list_HMC_1 + data_list_HMC_2 + data_list_HMC_3

data_root_arg = 'dataset/Argoverse'
data_list_arg = glob.glob(data_root_arg+'/val/data/*.csv')

x_tot = []
y_tot = []
for i in range(len(data_list_HMC)):
    data = data_list_HMC[i]
    with open(data, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if row[0].split(',')[2] == 'AV':
                x = float(row[0].split(',')[3])
                y = float(row[0].split(',')[4])
                x_tot.append(x)
                y_tot.append(y)

for i in range(len(data_list_arg)):
    data = data_list_arg[i]
    with open(data, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if row[0].split(',')[2] == 'AV':
                x = float(row[0].split(',')[3])
                y = float(row[0].split(',')[4])
                x_tot.append(x)
                y_tot.append(y)

plt.scatter(map_point[:,0], map_point[:,1])
plt.scatter(x_tot, y_tot)
plt.show()
plt.axis('equal')