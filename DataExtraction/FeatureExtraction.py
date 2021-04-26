# Extract feature data from pose data
import csv
import json
import math
import os

#path
#path = ["192_168_53_8a", "192_168_53_8b", "192_168_53_8c", "192_168_53_10a", "192_168_53_10b", "192_168_53_54", "192_168_53_55","192_168_53_56"]
path = ["192_168_53_8a"]
#ouput name
output = ["train_data1"]

for name in range(len(path)):
    files = os.listdir("../Data/pose/"+path[name])
    files.sort();

    with open(output[name]+".csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["descent_velocity", "angle_centerline", "human body's external rectangle", "class"])


    with open("../Data/label/"+path[name]+".csv", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)


    ans_corr = 0;

    for index in range(len(files)-1):

        if files[index].find(".json") != -1 and files[index+1].find(".json") != -1:
            with open("../Data/pose/"+path[name]+'/'+files[index], 'r') as f:
                video1 = json.load(f)

            with open("../Data/pose/"+path[name]+'/'+files[index+1], 'r') as f:
                video2 = json.load(f)

            if len(video1['people']) == 0 or len(video2['people']) == 0:
                with open(output[name]+".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["0",  "0",  "0", str(data[index + 1][1])])

                if int(data[index+1][1]) == 0:
                    ans_corr = ans_corr + 1
            else:
                list_keypoint_video1 = video1['people'][0]['pose_keypoints_2d']
                list_keypoint_video2 = video2['people'][0]['pose_keypoints_2d']

                # computing the speed of descent at the center of the hip joint
                yt1 = (list_keypoint_video1[28] + list_keypoint_video1[37]) / 2
                yt2 = (list_keypoint_video2[28] + list_keypoint_video2[37]) / 2
                descent_velocity = abs(yt1-yt2)/0.04



                # computing the human body centerline angle with the ground
                xt = (list_keypoint_video1[66]+list_keypoint_video1[57]) / 2
                yt = (list_keypoint_video1[67]+list_keypoint_video1[58]) / 2
                xt0 = list_keypoint_video1[0]
                yt0 = list_keypoint_video1[1]

                if xt0-xt == 0:
                    angle_centerline = math.pi/2
                else:
                    angle_centerline = math.atan(abs(yt0-yt)/abs(xt0-xt))

                # computing human body's external rectangle(width-to-height ratio)
                x_min = 65535
                x_max = 0
                y_min = 65535
                y_max = 0

                for i in range(len(list_keypoint_video1)):
                    if i%3 == 0:
                        if list_keypoint_video1[i] > x_max:
                            x_max = list_keypoint_video1[i]
                        if list_keypoint_video1[i] < x_min:
                            x_min = list_keypoint_video1[i]
                    if i%3 == 1:
                        if list_keypoint_video1[i] > y_max:
                            y_max = list_keypoint_video1[i]
                        if list_keypoint_video1[i] < y_min:
                            y_min = list_keypoint_video1[i]

                height = y_max - y_min
                width = x_max - x_min
                p = width/height

                # computing hand action
                x1 = list_keypoint_video1[0]

                ans = 0
                if descent_velocity > 0.0009 and angle_centerline < math.atan(1) and p > 1:
                    ans = 1

                with open(output[name]+".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(descent_velocity), str(angle_centerline), str(p), str(data[index + 1][1])])

                if int(data[index+1][1]) == ans:
                    ans_corr = ans_corr + 1








