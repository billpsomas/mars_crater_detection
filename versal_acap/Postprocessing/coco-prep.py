import sys
import json
model_name = sys.argv[1]

predictions = []

listed_lines = []
with open(f'output_boxes_{model_name}.txt', 'r') as f:
    for line in f:
        items = line.strip().split(" ")
        listed_lines.append(items)

tester = {}


for i in listed_lines:
    if i[0] == "Image":
        j = 4

        while (listed_lines[listed_lines.index(i)+j][0] != ''):
            predictions.append({"image_id": int(i[2][:-4]), "category_id" : 1 ,"bbox" : [round(float(listed_lines[listed_lines.index(i)+j][0]),5),round(float(listed_lines[listed_lines.index(i)+j][1]),5),round(float(listed_lines[listed_lines.index(i)+j][2]),5), round(float(listed_lines[listed_lines.index(i)+j][3]),5), round(float(listed_lines[listed_lines.index(i)+j][4]),5)]})
            j += 1

merged_dict = {}
for item in predictions:
    img_id = item['image_id']
    val = item['bbox']
    if img_id not in merged_dict:
        merged_dict[img_id] = []
    merged_dict[img_id].append(val)

# Step 2: Convert the dictionary back to a list of dictionaries
#output_list = [{'img_id': img_id, 'val': vals} for img_id, vals in merged_dict.items()]

#print(output_list)
#tester[int(i[2][:-4])] = [round(float(listed_lines[listed_lines.index(i)+j][0]),5),round(float(listed_lines[listed_lines.index(i)+j][1]),5),round(float(listed_lines[listed_lines.index(i)+j][2]) + float(listed_lines[listed_lines.index(i)+j][0]),5),round(float(listed_lines[listed_lines.index(i)+j][3])+float(listed_lines[listed_lines.index(i)+j][1]),5), round(float(listed_lines[listed_lines.index(i)+j][4]),5)]


with open("coco-preped-{}.json".format(model_name), 'w') as f:
    json.dump(merged_dict,f)