import os
import json

def read_txt(txt_path):
    '''read text files'''
    with open(txt_path) as f:
        lines = f.readlines()
    return lines

def collect_cords_txt(txt_path,imshape):
    '''read text files and convert with respect to imshape'''
    class_dict = {}
    xwd, ywd = imshape
    with open(txt_path) as f:
        lines = f.readlines()
        line_fl = [[float(r) for r in i.split(" ")] for i in lines.copy()]
        classes = [i[0] for i in line_fl.copy()]

    for i in set(classes.copy()):
        class_dict[i] = classes.count(i)

    new_line =[]
    for i in line_fl.copy():
        new_line.append([i[0], int(i[1]*xwd), int(i[2]*ywd), int(i[3]*xwd), int(i[4]*ywd)])

    return new_line, class_dict, line_fl


def save_json_topath(json_dict, json_path):
    json_object = json.dumps(json_dict, indent=4)

    # Writing to sample.json
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

def pathexist(path_fol):
    if not os.path.exists(path_fol):
        print("the path dont exist")
        raise Exception("Tha path doesnot exists")

def read_json(json_file):
    f = open(json_file)
    data = json.load(f)
    f.close()
    return data