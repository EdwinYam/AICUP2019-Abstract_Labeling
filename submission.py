import csv

import ast
label_type = ['all', 'background', 'objectives', 'methods', 'results', 'conclusions', 'others']
# label_type = ['all']

label_map = {'BACKGROUND_label':0, 'OBJECTIVES_label':1, 'METHODS_label':2, 'RESULTS_label':3, 'CONCLUSIONS_label':4, 'OTHERS_label':5, 'NONE_label':6}

predictions = dict()
raw_predictions = { l:list() for l in label_type }
for l in label_type:
    
    with open('preds/preds_public_{}.txt'.format(l)) as f:
        for line in f.readlines():
            raw_predictions[l].append(ast.literal_eval(line))
    for abstract in raw_predictions[l]:
        abstract_id = abstract[0]
        sent_labels = abstract[1]
        for idx, label in enumerate(sent_labels):
            sentence_id = abstract_id + '_S' + format(idx+1, '03d')
            if sentence_id not in predictions.keys():
                predictions[sentence_id] = [label_map[label]] if label_map[label] != 6 else list()
            elif label_map[label] != 6:
                predictions[sentence_id].append(label_map[label])
    
    
count = { i:0 for i in range(1,8) }
count_none = { i:0 for i in range(6)}
unmatch = 0
for labels in list(predictions.values()):
    if len(labels)>3:
        if labels[0] not in labels[1:]:
            print(labels)
    if len(labels)==1:
        count_none[labels[0]] += 1
    if len(labels)>=2:
        if labels[0] not in labels[1:]:
            unmatch += 1
    
    count[len(labels)] += 1

print(unmatch, count, count_none)

for sentence_id in list(predictions.keys()):
    if len(predictions[sentence_id]) > 1:
        predictions[sentence_id] = predictions[sentence_id][1:]
    #if predictions[sentence_id][-1] == 5  and len(predictions[sentence_id]) > 1:
    #    predictions[sentence_id] = [5]

    labels = predictions[sentence_id]
    if 5 in labels and len(labels) > 1:
        predictions[sentence_id] = predictions[sentence_id][:-1]

csvfile = open('final_submission.csv','w')
writer = csv.writer(csvfile)
#csvfile = open('task1_sample_submission.csv', 'r')
#rows = csv.reader(csvfile)
writer.writerow(['order_id', 'BACKGROUND','OBJECTIVES','METHODS','RESULTS','CONCLUSIONS','OTHERS'])

count= 1
for sentence_id in list(predictions.keys()):
    output = [ int(i in predictions[sentence_id]) for i in range(6) ]
    writer.writerow([sentence_id] + output)
    count += 1
    

#for idx, row in enumerate(rows):
#    if idx >= count:
#        writer.writerow(row)

predictions = dict()
raw_predictions = { l:list() for l in label_type }
for l in label_type:    
    with open('preds/preds_private_{}.txt'.format(l)) as f:
        for line in f.readlines():
            raw_predictions[l].append(ast.literal_eval(line))
    for abstract in raw_predictions[l]:
        abstract_id = abstract[0]
        sent_labels = abstract[1]
        for idx, label in enumerate(sent_labels):
            sentence_id = abstract_id + '_S' + format(idx+1, '03d')
            if sentence_id not in predictions.keys():
                predictions[sentence_id] = [label_map[label]] if label_map[label] != 6 else list()
            elif label_map[label] != 6:
                predictions[sentence_id].append(label_map[label])
    
    
count = { i:0 for i in range(1,8) }
count_none = { i:0 for i in range(6)}
unmatch = 0
for labels in list(predictions.values()):
    if len(labels)>3:
        if labels[0] not in labels[1:]:
            print(labels)
    if len(labels)==1:
        count_none[labels[0]] += 1
    if len(labels)>=2:
        if labels[0] not in labels[1:]:
            unmatch += 1
    
    count[len(labels)] += 1

print(unmatch, count, count_none)

for sentence_id in list(predictions.keys()):
    if len(predictions[sentence_id]) > 1:
        predictions[sentence_id] = predictions[sentence_id][1:]
    #if predictions[sentence_id][-1] == 5  and len(predictions[sentence_id]) > 1:
    #    predictions[sentence_id] = [5]

    labels = predictions[sentence_id]
    if 5 in labels and len(labels) > 1:
        predictions[sentence_id] = predictions[sentence_id][:-1]

for sentence_id in list(predictions.keys()):
    output = [ int(i in predictions[sentence_id]) for i in range(6) ]
    writer.writerow([sentence_id] + output)
