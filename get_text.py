import os
import codecs
import chardet
import sys
import jieba

def detect_file_encoding(file_path):
     f = open(file_path, 'r')
     data = f.read(200)
     predict = chardet.detect(data)
     f.close()
     return predict['encoding']

def get_file_content(file_path):
    f = open(file_path, 'r')
    return f.read()

# with code detact mode
def get_file_content_hw(file_path):
     file_encoding = detect_file_encoding(file_path)
     f = codecs.open(file_path, 'r', file_encoding, errors='ignore')
     return f.read()

def text_process(file_path):
    data = {"neg": [], "pos": []}
    for i in ["neg", "pos"]:
        path = os.path.join(file_path, i)
        documents = os.listdir(path)
        for j in documents:
            line = get_file_content(os.path.join(path, j))
            data[i].append(list(jieba.cut(line.strip().split()[0], cut_all=False)))
        
        with open(file_path+"_"+i+".txt", 'w') as outfile:
            for line in data[i]:
                outfile.write(" ".join(line).encode('utf-8') + '\n')
    

if __name__ == '__main__':
    file_path = sys.argv[1]
    text_process(file_path)
