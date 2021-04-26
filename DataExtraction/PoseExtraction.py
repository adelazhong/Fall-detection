#Extract pose data to csv
import csv
import json
import math
import os

list_name = []
for i in range(75):
    if i%3 == 0:
        list_name.append("x"+str(int(i/3)))
    elif i % 3 == 1:
        list_name.append("y" + str(int(i / 3)))
    elif i % 3 == 2:
        list_name.append("c" + str(int(i / 3)))
list_name.append("class")




#path
#path = ["192_168_53_8a", "192_168_53_8b", "192_168_53_8c", "192_168_53_10a", "192_168_53_10b", "192_168_53_54", "192_168_53_55","192_168_53_56"]
path = ["192_168_53_8a"]

#output name
output = ["train_data2"]

for name in range(len(path)):
    files = os.listdir("../Data/pose/"+path[name])
    files.sort();

    with open(output[name]+".csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list_name)

    with open("../Data/label/"+path[name]+".csv", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    for index in range(len(files)):


        if files[index].find(".json") != -1:
            with open("../Data/pose/"+path[name]+'/'+files[index], 'r') as f:
                video1 = json.load(f)

            if len(video1['people']) == 0:
                list_content = []
                for i in range(75):
                    list_content.append("0")
                list_content.append(str(data[index + 1][1]))

                with open(output[name]+".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list_content)
            else:
                list_keypoint_video1 = video1['people'][0]['pose_keypoints_2d']
                list_content = []
                for i in range(len(list_keypoint_video1)):
                    list_content.append(list_keypoint_video1[i])
                list_content.append(str(data[index + 1][1]))
                with open(output[name]+".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list_content)
