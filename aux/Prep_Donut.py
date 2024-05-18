import json, os, random, sys
import psutil

DONUT_ANN_PATH = {
    'val': 'data/anns/val/',
    'test': 'data/anns/test/',
    'train': 'data/anns/train/'   
}


def tokens_to_string(cell):
    return ''.join(cell['tokens'])


def get_span_type(coords, max_tup):
    
    i, j = coords
    max_row, max_col = max_tup
    
    neigs = {'up': "0",
            'right': "0",
            'down': "0",
            'left': "0"
           }
    
    if i > 0:
        neigs['up'] = "1"
        if i < max_row:
            neigs['down'] = "1"
    elif max_row != 0:
        neigs['down'] = "1"
    
    if j > 0:
        neigs['left'] = "1"
        if j < max_col:
            neigs['right'] = "1"
    elif max_col != 0:
        neigs['right'] = "1"
    
    return "span_type=" + neigs['right'] + neigs['down'] + neigs['up'] + neigs['left']



def create_cell(table, i, j, header, content, content_holder, span_type = "span_type=0000"):    
    aux_cell = {
        "row": i,
        "col": j,
        "col_header": header,
        "row_header": False,
        "span_type": span_type,
        "content_holder": content_holder,
        "content": ""
    }
    
    
    aux_content = tokens_to_string(content)
    aux_cell["content"] = aux_content
        
    table[i][j] = aux_cell 


def create_cells(table, start_i, start_j, header, span_tup, content):
    rowspan, colspan = span_tup
    max_i = start_i + rowspan
    max_j = start_j + colspan
    content_holder = False
    
    for i in range(start_i, max_i):
        for j in range(start_j, max_j):
            #if(i == (start_i + max_i)//2) and (j == (start_j + max_j)//2):
            if i == max_i-1 and j == max_j-1:
                content_holder = True
            else:
                content_holder = False
            create_cell(table, i, j, header, content, content_holder, get_span_type((i - start_i, j - start_j), (rowspan-1, colspan-1)))
             
    
    return max_j


def decode_span(rowspan, colspan, tag):
    if tag[:4] == " row":
        return "row", int(tag.split('"')[1])
    elif tag[:4] == " col":
        return "col", int(tag.split('"')[1])
    else:
        print(tag)
        raise unk


def crop_table(table, max_tuple):
    max_row, max_col = max_tuple
    
    new_table = [[] for i in range(max_row)]
    for i in range(max_row):
        header = False
        for j in range(max_col):
            if(table[i][j] == None):
                new_table[i].append({
                    "row": i,
                    "col": j,
                    "col_header": header,
                    "row_header": False,
                    "span_type": "span_type=0000",
                    "content_holder": True,
                    "content": ""
                })
                continue
            elif(table[i][j] != None and table[i][j]['col_header']):
                header = True
            
            new_table[i].append(table[i][j])
            
    return new_table



def json_to_ann(line):
    annotation = json.loads(line)
    
    span = False
    
    header = False
        
    max_row = 0
    max_col = 0
        
    rowspan = 1
    colspan = 1
        
    row_i = 0
    col_i = 0
    content_i = 0

    table = [[] for i in range(100)]
    for i in range(100):
        table[i] = [None for j in range(100)]
        
    
    ann_html = annotation['html']

    for html_tag in ann_html['structure']['tokens']:
        match html_tag:
            case "<td>":
                while(table[row_i][col_i] != None):
                    col_i += 1
                create_cell(table, row_i, col_i, header, ann_html['cells'][content_i], True)
                col_i += 1
            case "<td":
                pass
            case ">":
                while(table[row_i][col_i] != None):
                    col_i += 1
                col_i = create_cells(table, row_i, col_i, header, (rowspan, colspan), ann_html['cells'][content_i])
            case "</td>":
                content_i += 1
                rowspan = 1
                colspan = 1
            case "<thead>":
                header = True
            case "</thead>":
                header = False
            case "<tbody>":
                pass
            case "</tbody>":
                pass
            case "<tr>":
                col_i = 0
            case "</tr>":
                row_i += 1
                max_row = max(max_row, row_i)
                max_col = max(max_col, col_i)
            case _:
                span_coord, span_size = decode_span(rowspan, colspan, html_tag)
                
                if(span_coord == "row"):
                    rowspan = span_size
                else:
                    colspan = span_size
    
    return crop_table(table, (max_row, max_col)), annotation['filename'], annotation['split']



def json_to_html(line):
    annotation = json.loads(line)        

    seq = ""
    content_i = 0
    
    ann_html = annotation['html']

    for html_tag in ann_html['structure']['tokens']:
        seq += html_tag
        match html_tag:
            case "<td>":
                seq += tokens_to_string(ann_html['cells'][content_i])
            case ">":
                seq += tokens_to_string(ann_html['cells'][content_i])
            case "</td>":
                content_i += 1
            case _:
                pass
    
    return seq




from tqdm.auto import tqdm

with open("data/PubTabNet_2.0.0.jsonl", encoding="utf-8") as f:
    for line in tqdm(f):
        table, file, split = json_to_ann(line)
        with open(DONUT_ANN_PATH[split] + file[:-4] +".json", 'w') as out:
            json.dump({'tables': [table]}, out, ensure_ascii=False, indent=4)

        html = json_to_html(line)
        with open(DONUT_ANN_PATH[split] + file[:-4] +"-HTML.json", 'w') as out:
            json.dump(html, out, ensure_ascii=False, indent=4)





