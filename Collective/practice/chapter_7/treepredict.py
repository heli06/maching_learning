#!/usr/bin/env python
# coding: utf-8

# In[1]:


my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]


# In[2]:


class decisionnode:
    # col 待检验的判断条件所对应的列索引值
    # value 为了使结果为true，当前列必须匹配的值
    # tb 结果为true时，对应的节点
    # fb 结果为false时，对应的节点
    # results 当前分支的结果，是一个字典。除叶节点外，其他节点上该值为None
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        
        

# 在某一列上对数据集合进行拆分，能够处理数据型数据或名词型数据
# raw 列表
# column 表中列所在位置的数字
# value 用以对列进行拆分的参考值
def divideset(rows, column, value):
    # 定义一个函数，令其告诉我们数据行属于第一组（返回true）还是第二组（返回false）
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # 将数据集拆分成两个集合，并返回
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

# 对各种可能的结果进行计数（每一行数据的最后一列记录了这一计数结果）
def uniquecounts(rows):
    results = {}
    for row in rows:
        # 计数结果在最后一列
        r = row[len(row)-1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results

# 基尼不纯度 随机放置的数据项出现于错误分类中的概率
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1])/total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2])/total
            imp += p1*p2
    return imp

# 熵，代表集合的无序程度
# 熵在这里是遍历所有可能的结果之后，所得到的p(x)log(p(x))之和
def entropy(rows):
    from math import log
    log2 = lambda x: log(x)/log(2)
    results = uniquecounts(rows)
    # 此处开始计算熵的值
    ent = 0.0
    for r in results.keys():
        p = float(results[r])/len(rows)
        ent = ent-p*log2(p)
    return ent


# In[3]:


# 分割集合
divideset(my_data, 2, 'yes')


# In[4]:


print(giniimpurity(my_data))
print(entropy(my_data))
set1, set2 = divideset(my_data, 2, 'yes')
print(giniimpurity(set1))
print(entropy(set1))


# In[5]:


# 建树
def buildtree(rows, scoref = entropy):
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)
    # 定义一些变量以记录最佳拆分条件
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    
    column_count = len(rows[0])-1
    for col in range(0, column_count):
        # 在当前列中生成一个由不同值构成的序列
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # 接下来根据这一列中的每个值，尝试对数据集进行拆分
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)
            
            # 信息增益
            p = float(len(set1))/len(rows)
            gain = current_score - p*scoref(set1) - (1-p)*scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
                
    # 创建子分支
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


# In[6]:


tree = buildtree(my_data)


# In[7]:


# 打印树
def printtree(tree, indent=''):
    # 这是一个叶节点吗？
    if tree.results != None:
        print(indent + str(tree.results))
    else:
        # 打印判断条件
        print(indent + str(tree.col) + ':' + str(tree.value) + '? ')
        # 打印分支
        print(indent + 'T->')
        printtree(tree.tb, indent + '  ')
        print(indent + 'F->')
        printtree(tree.fb, indent + '  ')


# In[8]:


printtree(tree)


# In[9]:


def getwidth(tree):
    if tree.tb == None and tree.fb == None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)

def getdepth(tree):
    if tree.tb == None and tree.fb == None:
        return 0
    return max(getdepth(tree.fb), getdepth(tree.fb)) + 1


# In[10]:


from PIL import Image, ImageDraw


# In[11]:


# 绘制
def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 320
    
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)
    
    drawnode(draw, tree, w/2, 20)
    img.save(jpeg, 'JPEG')

def drawnode(draw, tree, x, y):
    if tree.results == None:
        # 得到每个分支的宽度
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100
        # 确定此节点所要占据的总空间
        left = x - (w1+w2)/2
        right = x + (w1+w2)/2
        # 绘制判断条件字符串
        draw.text((x-20, y-10), str(tree.col) + ':' + str(tree.value), (0,0,0))
        # 绘制到分支的连线
        draw.line((x, y, left+w1/2, y+100), fill=(255, 0, 0))
        draw.line((x, y, right-w2/2, y+100), fill=(255, 0, 0))
        # 绘制分支的节点
        drawnode(draw, tree.fb, left+w1/2, y+100)
        drawnode(draw, tree.tb, right-w2/2, y+100)
    else:
        txt = '\n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x-20, y), txt, (0,0,0))


# In[12]:


drawtree(tree, jpeg='treeview.jpg')


# In[13]:


# 分类
def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
    return classify(observation, branch)


# In[14]:


classify(['(direct)', 'USA', 'yes', 5], tree)


# In[15]:


# 剪枝
def prune(tree, mingain):
    # 如果分支不是叶节点，进行剪枝操作
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)
    
    # 如果两个子分支都是叶子结点，判断它们是否需要合并
    if tree.tb.results != None and tree.fb.results != None:
        # 构造合并后的数据集
        tb, fb = [], []
        for v,c in tree.tb.results.items():
            tb += [[v]]*c
        for v,c in tree.fb.results.items():
            fb += [[v]]*c
        
        # 检查熵的减少情况
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb))/2
        if delta < mingain:
            # 合并分支
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)


# In[16]:


prune(tree, 0.1)
printtree(tree)
drawtree(tree, jpeg='treeview_1.jpg')


# In[17]:


prune(tree, 1.0)
printtree(tree)
drawtree(tree, jpeg='treeview_2.jpg')


# In[18]:


# 带权重的分类
def mdclassify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        if v == None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k,v in tr.items():
                result[k] = v*tw
            for k,v in fr.items():
                if k not in result:
                    result[k] = 0
                result[k] += v*fw
            return result
        
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


# In[19]:


print(mdclassify(['google', None, 'yes', None], tree))
print(mdclassify(['google', 'France', 'None', 'None'], tree))


# In[ ]:


# 以方差作为衡量标准
def variance(rows):
    if len(rows) == 0:
        return 0
    data = [float(row[len(row)-1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d-mean) ** 2 for d in data]) / len(data)
    return variance

