import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import csv

matplotlib.use('Qt5Agg')
tree = ET.parse('dataset/HMC/map_files/pruned_argoverse_HMC_10317_vector_map.xml')
root = tree.getroot()

map_point = []
for child in root:
    if child.tag == 'node':
        pt = [float(child.attrib['x']), float(child.attrib['y'])]
        map_point.append(pt)
map_point = np.array(map_point)

data_root = 'dataset/HMC'
data_list_1 = glob.glob(data_root+'/train/data/*.csv')
data_list_2 = glob.glob(data_root+'/test_obs/data/*.csv')
data_list_3 = glob.glob(data_root+'/val/data/*.csv')
data_list = data_list_1 + data_list_2 + data_list_3

x_tot = []
y_tot = []
for i in range(len(data_list)):
    data = data_list[i]
    with open(data, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if row[0].split(',')[1] == '000-0000-0000':
                x = float(row[0].split(',')[3])
                y = float(row[0].split(',')[4])
                x_tot.append(x)
                y_tot.append(y)


plt.scatter(map_point[:,0], map_point[:,1])
plt.scatter(x_tot, y_tot)
plt.show()