
def read_a_file(targetFile):
    """
    read one file of train or test file
    :param tragetFile: the target file that we want get
    :return: the content of targetFile. The type of content is list.
    """
    file = open(targetFile,"r",encoding = "utf-8")
    content = []
    #一句一句地读
    while 1:
        if(file.readline() == ''):      #read <text>
            break
        content.append(file.readline().rstrip().split())
        file.readline()                 #read </text>
    return content

def get_all_trainData(targetFiles):
    allDatas = []
    for i in range(len(targetFiles)):
        allDatas.append(read_a_file(targetFiles[i]))
    return allDatas

def get_stopWords(targetFile):
    file = open(targetFile,"r",encoding = "utf-8")
    stopWords = []
    while 1:
        temp = file.readline()
        if temp == '':
            break
        stopWords.append(temp.rstrip())
    return stopWords

"""content = read_a_file('../Data/Tsinghua/train/体育.txt')
print(content[0][7])"""

"""stopWs = get_stopWords('../Data/Tsinghua/stop_words_zh.txt')
if '?' in stopWs:
    print("good")"""
