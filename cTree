def dTree(data,header=True,config=None,ops=[1,0.01],method=1):
    data,feats=buildFeat(data,header,config)
    tree=crateTree(data,feats,ops,method)
    return(tree)
def crateTree(data,feats,ops,method):
    sd={}
    n=len(data)
    for item in data:
        sd[item[0]]=sd.get(item[0],0)+1
    for item in sd:
        sd[item]=sd[item]/n
    bFeat,bVal,bEntropy=bestSplit(data,feats,ops,method)
    entro=entropy(data)
    if bFeat==None:
        tree={
            'cond':'leaf',
            'acc':sd,
            'n':n,
            'entropy':entro
            }
    else:
        if feats[bFeat]["value"]!=None:
            subD1,subD2=splitData(data,bFeat,bVal,False)
        else:
            subD1,subD2=splitData(data,bFeat,bVal,True)
        tree={
            'cond':'branch',
            'acc':sd,
            'n':n,
            'entropy':entro,
            'feat':feats[bFeat]["name"],
            'val':bVal,
            'left':crateTree(subD1,feats,ops,method),
            'right':crateTree(subD2,feats,ops,method)
            }
    return(tree)

def buildFeat(data1,header=True,config=None):
    feat={}
    if header==True:
        tempt=data1[0]
    data=data1[1:]
    for i in range(len(tempt)):
        if i in config:
            feat[i]={"name":tempt[i],"value":None}
            for i1 in range(len(data)):
                data[i1][i]=float(data[i1][i])
            continue
        value=list(map(lambda x:x[i],data))
        val=unique_new(value)
        feat[i]={"name":tempt[i],"value":val}
    return(data,feat)

def unique(value):
    from cain import qsort
    qsort(value,0,len(value))
    var=[]
    uni=None
    for i in range(len(value)):
        if value[i]!=uni:
            uni=value[i]
            var.append(uni)
    return(var)

def unique_new(data):
    from cain import qsort
    s=[]
    for i in range(len(data)):
	    if data[i] not in s:
		    s.append(data[i])
    qsort(s,0,len(s))
    return(s)

def splitData(rawdata,colId,value,continu=False):
    if continu==False:
        subD1=list(filter(lambda x:x[colId]==value,rawdata))
        subD2=list(filter(lambda x:x[colId]!=value,rawdata))
    else:
        subD1=list(filter(lambda x:x[colId]<=value,rawdata))
        subD2=list(filter(lambda x:x[colId]>value,rawdata))
    return subD1,subD2

def entropy(data):
    from math import log
    n=len(data)
    sd={}
    s=0.0
    for item in data:
        sd[item[0]]=sd.get(item[0],0)+1
    for item in sd:
        s-=(sd[item]/n)*log((sd[item]/n),2)
    return(s)

def bestSplit(data,feats,ops=[1,0.01],method=1):
    from cain import qsort
    bFeat=None
    bVal=None
    bEntropy=ops[1]
    sEntropy=entropy(data)
    for feat in feats:
        if feat==0:
            continue
        if feats[feat]["value"]==None:
            l=list(map(lambda x:x[feat],data))
            i=0
            k=l[0]-1
            while i<len(l):
                if l[i]>k:
                    k=l[i]
                    entro=getBfeat(data,feat,l[i],bEntropy,sEntropy,ops[0],True)
                    if entro != None:
                        bFeat=feat
                        bVal=k
                        bEntropy=entro
                i+=1
            continue
        else:
            for val in feats[feat]["value"]:
                entro=getBfeat(data,feat,val,bEntropy,sEntropy,ops[0],False)
                if entro != None:
                    bFeat=feat
                    bVal=val
                    bEntropy=entro
    return bFeat,bVal,bEntropy


def getBfeat(data,feat,value,oldEn,sEntropy,ops,continual,method=1):
    vVal=None
    subD1,subD2=splitData(data,feat,value,continual)
    if len(subD1)>=ops and len(subD2)>=ops:
        entro=evalEntropy(subD1,subD2,sEntropy,method)
        if entro>oldEn:
            vVal=entro
    return(vVal)
    
                
        
def evalEntropy(subD1,subD2,sEntropy,method):
    from math import log
    a=len(subD1)
    b=len(subD2)
    c=a+b
    return((sEntropy-entropy(subD1)*a/c-entropy(subD2)*b/c)/(-log(a/c,2)*a/c-log(b/c,2)*b/c))
