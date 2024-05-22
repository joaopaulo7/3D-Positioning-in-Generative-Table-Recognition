from transformers import DonutProcessor


class TabeleiroProcessor(DonutProcessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_types = ["<cell>", "<col_header>", "<row_header>", "<row_and_col_header>"]
        self.content_types = ["<content_row_and_col_header>", "<content_row_header>", "<content_col_header>", "<content>"]

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.cell_types.append("<span_type=0" + str(i) + str(j) + str(k) + ">")
                    self.cell_types.append("<span_type=1" + str(i) + str(j) + str(k) + ">")
        
        self.tokenizer.add_tokens(["<table_extraction>", "<table>", "<row>"] + self.content_types + self.cell_types)
    
    
    #ANNOTATION TO SEQUENCE
    def _cel2token(self, cell):
        if cell['span_type'][10:] != '0000':
            sequence = "<" + cell['span_type'] + ">"
            if cell['content_holder']:
                if cell['row_header'] and cell['col_header']:
                    sequence += "<content_row_and_col_header>"
                elif cell['col_header']:
                    sequence += "<content_col_header>"
                elif cell['row_header']:
                    sequence += "<content_row_header>"
                else:
                    sequence += "<content>"
                sequence += cell['content']
        else:
            sequence = ""
            if cell['content_holder']:
                if cell['row_header'] and cell['col_header']:
                    sequence += "<row_and_col_header>"
                elif cell['col_header']:
                    sequence += "<col_header>"
                elif cell['row_header']:
                    sequence += "<row_header>"
                else:
                    sequence += "<cell>"
                sequence += cell['content']
            
        return sequence

    def _row2token(self, row):
        sequence = "<row>"
        for cell in row:
            sequence += self._cel2token(cell)
        
        return sequence


    def _table2token(self, table):
        sequence = "<table>"
        for row in table:
            sequence += self._row2token(row)
        
        return sequence


    def json2token(self, json):
        sequence = ""
        if('tables' in json):
            for table in json['tables']:
                sequence += self._table2token(table)

        return sequence
    
    
    #SEQUENCE TO ANNOTATION
    
    def _token2cell(self, seq, i, cell_coord, cell_type):
        if cell_type[:5] == "<span": 
            span_type = cell_type[1:-1]
            cell_type = self.decode(seq[i])
            if cell_type[:5] == "<cont":
                i += 1
                if cell_type != "<content>":
                    cell_type = "<"+cell_type[9:]
                else:
                    cell_type = "<cell>"
        else:
            span_type = "span_type=0000"
        
        x, y = cell_coord
        aux_cell = {
            "row": x,
            "col": y,
            "col_header": cell_type in ["<col_header>", "<row_and_col_header>"],
            "row_header": cell_type in ["<row_header>", "<row_and_col_header>"],
            "colspan": -1 if span_type != "span_type=0000"  else 1,
            "rowspan": -1 if span_type != "span_type=0000"  else 1,
            "span_type": span_type,
            "content": ""
        }
        start_cont = i
        while self.decode(seq[i]) not in ["<row>", "</s>", "<table>"] + self.cell_types:
            i += 1
            
        aux_cell['content'] += self.decode(seq[start_cont: i])
        
        return aux_cell, i

    def _token2row(self, seq, i, row_id):
        cells = []
        while self.decode(seq[i]) not in ["<row>", "</s>", "<table>"]:
            if self.decode(seq[i]) in self.cell_types:
                aux_cell, i = self._token2cell(seq, i+1, (row_id, len(cells)), self.decode(seq[i]))
                cells.append(aux_cell)
            else:
                i += 1
        return cells, i


    def _token2table(self, seq, i):
        rows = []
        while self.decode(seq[i]) not in ["</s>", "<table>"]:
            if self.decode(seq[i]) == "<row>":
                aux_row, i = self._token2row(seq, i+1, len(rows))
                rows.append(aux_row)
            else:
                i += 1
                
        return rows, i


    def token2ann(self, seq, i):
        tables = []
        while self.decode(seq[i]) != "</s>":
            if self.decode(seq[i] == "<table>"):
                aux_table, i = self._token2table(seq, i+1)
                #self._crop_empty_left(aux_table)
                
                self._define_spannings(aux_table)
                
                tables.append(aux_table)
                
            else:
                i += 1
        return {'tables': tables}

    #SEQ TO ANN AUX    

    def _crop_empty_left(self, table):
        for row in table:
            empty_left = []
            for cell in row:
                if(cell['content'] == "" and cell['colspan'] == 1):
                    empty_left.append(cell)
                else:
                    empty_left.clear()
            
            for cell in empty_left:
                cell['colspan'] = 0
                cell['rowspan'] = 0


    def _update_vals(self, cell, content, col_header, row_header):
        cell['colspan'] = 0
        cell['rowspan'] = 0
        
        if cell['content'] != "":
            content = cell['content']
        
        col_header = col_header or cell['col_header']
        row_header = row_header or cell['row_header']
        
        return content, col_header, row_header


    def _define_by_path(self, cell, table):
        col_header, row_header = False, False
        i, j = cell['row'], cell['col']
        first_j = j

        content = ''
        
        while True:
            while True:
                content, col_header, row_header = self._update_vals(cell, content, col_header, row_header)
                
                if(cell['span_type'][-4:-3] != '1') or len(table) <= i or len(table[i]) <= j:
                    break
                j += 1
                cell = table[i][j]
            
            if(cell['span_type'][-3:-2] != '1'): #case it's the end of cell
                break
            i += 1
            j = first_j
            
            if len(table) <= i or len(table[i]) <= j: #case the  sequence is broken
                break
            cell = table[i][j] # oiii =P
        
        return (i, j), content, col_header, row_header

        
    def _define_spannings(self, table):
        for row in table:
            for cell in row:
                if(cell['colspan'] == -1):
                    end_coord = (cell['row'], cell['col'])
                    content = cell['content']
                 
                    end_coord, content, col_header, row_header = self._define_by_path(cell, table)
                    
                    
                    cell['rowspan'] = end_coord[0] - cell['row'] + 1
                    cell['colspan'] = end_coord[1] - cell['col'] + 1
                    cell['col_header'] = col_header
                    cell['row_header'] = row_header
                    cell['content'] = content
    
    # ANN TO HTML
    
    def _cell2html(self, cell):
        seq = "<td"
        if(cell['rowspan'] > 1):
            seq += ' rowspan="' + str(cell['rowspan']) +'"'
        if(cell['colspan'] > 1):
            seq += ' colspan="' + str(cell['colspan']) +'"'
        seq += ">" + cell["content"] + "</td>"
        
        return seq

    def table2html(self, table):
        seq = ""
        head = False
        body = False
        for row in table:
            row_seq = ""
            count_header = 0
            row_len = len(row)
            for cell in row:
                if(cell['colspan'] > 0):
                    count_header += cell['col_header']
                    row_seq += self._cell2html(cell)
                else:
                    row_len -= 1
                
            if(count_header  > row_len/2):
                if body:
                    seq += "</tbody>"
                    body = False
                    
                if not head:
                    seq += "<thead>"
                    head = True
                seq +=  "<tr>" + row_seq + "</tr>" 
            else:
                if head:
                    seq += "</thead>"
                    head = False
                    
                if not body:
                    seq += "<tbody>"
                    body = True
                    
                seq +=  "<tr>" + row_seq + "</tr>"
        
        if head:
            seq += "</thead>"
            
        if body:
            seq += "</tbody>"
        
        return seq
