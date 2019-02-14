#!usr/bin/env python3
import numpy as np
from numpy.linalg import solve
from matplotlib import pyplot as plt
import math
import cv2 as cv2
import random

import copy

# 读取文件
original = cv2.imread('/home/hurly/Desktop/diffcult1.jpg')
# 噪声处理
filterimg = cv2.GaussianBlur(original, (5, 5), 0)
black = np.zeros((2000, 2000, 3), dtype=np.uint8)
blacktemp1 = np.zeros((2000, 2000, 3))
# bgr--->rgb
rgb = cv2.cvtColor(filterimg, cv2.COLOR_BGR2RGB)
# 转化为灰度
gray = cv2.cvtColor(filterimg, cv2.COLOR_BGR2GRAY)
# 转化为二值图像
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# # 填充图像
# temp=thresh.copy()
# h,w=thresh.shape[:2]
# flag=8|(255<<8)
# mask=np.zeros([h+2,w+2],np.uint8)
# cv2.floodFill(temp,mask,(0,0),255,0,0,flag)
# # ?需要剪裁否                          ?
# fillimg=thresh+(~temp
# 寻找边界
thresh, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(original, contours, -1,(0,255,0) , 5)
# cv2.namedWindow("default",0)
# cv2.resizeWindow("default",100,100)
# cv2.imshow('default', original)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# process contour
contourtemp = list()
for contour in contours:
    # print(cv2.contourArea(contour))
    if(cv2.contourArea(contour) > 1000):
        contourtemp.append(contour)
contours = contourtemp
for contour in contours:
    print(contour.shape)
# cv2.drawContours(black, contours, -1, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 3)
# plt.imshow(black)
# plt.show()
# originalcopy=original.copy()
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(originalcopy,(x,y),5,(255,0,0),-1)

# print("有%d个轮廓" % len(contours))
# # 画轮廓
# cv2.drawContours(originalcopy, contours, -1, (0, 255, 0), 3)
# cv2.namedWindow("default",0)
# cv2.setMouseCallback("default",draw_circle)
# cv2.resizeWindow("default",100,100)
# while(1):
#     cv2.imshow('default',originalcopy)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()


# # search for corner
# print("search for corners")
# black=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
# black=cv2.cornerHarris(black,2,3,0.01)
# black=cv2.dilate(black,None)
# original[black>0.1*black.max()]=[0,0,255]
# cv2.imshow('gray',original)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 多边形近
while True:
    x = input("please input the approximate degree  ")
    x = float(x)
    if(x < 0):
        break
    print("%d" % x)
    originalcopy = original.copy()
    contourscopy = contours.copy()
    for i in range(len(contourscopy)):
        contourscopy[i] = cv2.approxPolyDP(contourscopy[i], x, True)
        print(contourscopy[i].shape)
# draw the point
    for contour in contourscopy:
        print(contour.shape)
        cv2.drawContours(originalcopy, contour, -1, (0, 255, 0), 8)
    cv2.namedWindow("default", 0)
    cv2.resizeWindow("default", 100, 100)
    cv2.imshow('default', originalcopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# print("轮廓坐标")
# print(contours[0])
# # 判断凹点凸点
# hull=cv2.convexHull(contours[0],clockwise=True,returnPoints=False)
# print("凸点")
# print(hull)

# show each piece
i=1
for contour in contours:
    blacktemp=blacktemp1.copy()
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    color=(0,125,125)
    cv2.drawContours(blacktemp, contours,i-1,color, 10)
    plt.subplot(2,(len(contours)+1)/2,i)
    plt.imshow(blacktemp)
    plt.title('contour%d'%(i-1))
    plt.xticks([]),plt.yticks([])
    i+=1
plt.show()

contours = contourscopy

# contourtopic=np.empty(shape=original.shape,dtype=np.int)
# for contour in contours:
#     cv2.drawContours(contourtopic,[contour],0,(0,255,0),5)


class vertex:
    "顶点信息"

    def __init__(self, prepoint, lasside, nexside, angle):
        self.prepoint = prepoint
        self.lasside = lasside
        self.nexside = nexside
        self.angle = angle

    def __str__(self):
        return '("顶点"(%d,%d)"上一条边"%d"下一条边"%d"角度"%d)' % (self.prepoint[0], self.prepoint[1], self.lasside, self.nexside, self.angle)


# 测试vertex信息


def test_vertex():
    vertex1 = vertex([2, 3], 10, 21, 90)
    print(vertex1)


class contourInfoEach:
    def __init__(self, contourInfoNode, vertexNum):
        self.contourInfoNode = contourInfoNode
        self.vertexNum = vertexNum


print("测试vertex信息")
test_vertex()
# 计算边长和角度
angleInfo = list()
contourInfo = list()
for i in range(len(contours)):
    contour = contours[i]
    contourInfoNode = list()
    vertexNum = contour.shape[0]
    temp_length = list()
    for j in range(vertexNum):  # 计算边长
        prepoint = contour[j][0]
        nextpoint = contour[(j+1) % vertexNum][0]
        length = math.sqrt((nextpoint[0]-prepoint[0])
                           ** 2+(nextpoint[1]-prepoint[1])**2)
        # approximate
        if(length < 10):
            laspoint = contour[(j-1) % vertexNum][0]
            nexnexpoint = contour[(j+2) % vertexNum][0]
            lasdiffer = prepoint-laspoint
            nexdiffer = nexnexpoint-nextpoint
            a = np.mat([[1/lasdiffer[0], -1/lasdiffer[1]],
                        [1/nexdiffer[0], -1/nexdiffer[1]]])
            b = np.mat([(prepoint[0]/lasdiffer[0]-prepoint[1]/lasdiffer[1]),
                        (nextpoint[0]/nexdiffer[0]-nextpoint[1]/nexdiffer[1])]).T
            x = solve(a, b)
            x = np.asarray(x)
            nextpoint = np.array([int(x[0][0]), int(x[1][0])])
            contour[(j+1) % vertexNum][0] = nextpoint
            length = math.sqrt(
                (nextpoint[0]-prepoint[0])**2+(nextpoint[1]-prepoint[1])**2)
            if(j == vertexNum-1):
                prepoint = nextpoint
                nextpoint = contour[1][0]
                lengthfir = math.sqrt(
                    (nextpoint[0]-prepoint[0])**2+(nextpoint[1]-prepoint[1])**2)
                temp_length[0] = lengthfir
        temp_length.append(length)
        # print(temp_length[j])
    hull = cv2.convexHull(contours[i], clockwise=True, returnPoints=False)
    contournew = list()
    keptfirst = 0
    # error handle
    for j in range(vertexNum):
        lastpoint = contour[(j-1) % vertexNum][0]
        prepoint = contour[j][0]
        nextpoint = contour[(j+1) % vertexNum][0]
        lastdiffer = prepoint-lastpoint
        nextdiffer = nextpoint-prepoint
        product = lastdiffer[0]*nextdiffer[0]+lastdiffer[1]*nextdiffer[1]
        angle = product/(temp_length[(j-1) % vertexNum]*temp_length[j])
        # if(angle>1 or angle<-1):
        #     print("angle is out of domain")
        angle = int(math.acos(angle)*57.3)
        if j in hull:
            angle = 180-angle
        else:
            angle = 180+angle

        if((angle > 175 and angle < 185)or temp_length[j] < 4):
            if(not contourInfoNode):
                keptfirst += temp_length[j]
            else:
                contourInfoNode[-1].nexside += temp_length[j]
        else:
            contourInfoNode.append(vertex(prepoint, int(
                temp_length[(j-1) % vertexNum]), int(temp_length[j]), angle))
        # angleInfo.append([angle, i, j])  # 存储角度位置信息
    # calculate angle and length
    vertexNum = len(contourInfoNode)
    for j in range(vertexNum):
        lastpoint = contourInfoNode[(j-1) % vertexNum].prepoint
        prepoint = contourInfoNode[j].prepoint
        nextpoint = contourInfoNode[(j+1) % vertexNum].prepoint
        lastdiffer = prepoint-lastpoint
        nextdiffer = nextpoint-prepoint
        product = lastdiffer[0]*nextdiffer[0]+lastdiffer[1]*nextdiffer[1]
        laslength = math.sqrt(lastdiffer[0]**2+lastdiffer[1]**2)
        nexlength = math.sqrt(nextdiffer[0]**2+nextdiffer[1]**2)
        angle_before = contourInfoNode[j].angle
        angle = product/(laslength*nexlength)
        # if(angle>1 or angle<-1):
        #     print("angle is out of domain")
        angle = int(math.acos(angle)*57.3)
        if(angle_before < 180):
            angle = 180-angle
        else:
            angle = 180+angle
        contourInfoNode[j] = vertex(prepoint, laslength, nexlength, angle)

    contourInfo.append(contourInfoEach(contourInfoNode, len(contourInfoNode)))

    # redraw contour
contoursn = list()
for contourInfoEach in contourInfo:
    contourInfoNode = contourInfoEach.contourInfoNode
    contourn = np.empty(shape=(len(contourInfoNode), 1, 2), dtype=np.int32)
    for i in range(len(contourInfoNode)):
        contourn[i][0] = contourInfoNode[i].prepoint
    contoursn.append(contourn)


# draw the point
originalcopy = original.copy()
for contour in contoursn:
    print(contour.shape)
    cv2.drawContours(originalcopy, contour, -1, (0, 255, 0), 8)
cv2.namedWindow("default", 0)
cv2.resizeWindow("default", 100, 100)
cv2.imshow('default', originalcopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("测试函数")
# def test(str):
#     return str+" i 'm fine"
# print(test("hello"))
# print("测试类")
# class test:
#     def __init__(self,name):
#         self.name=name
# a=test("hh")
# print(a)
# def testi():
#     for i in range(2):
#         print(i)
#     j=i+1
#     print("打印j的值",j)
# testi()
# angleInfo = sorted(angleInfo, key=lambda x: x[0])
# print("轮廓信息")
# for i in range(len(contourInfo[1].contourInfoNode)):
#     print(contourInfo[1].contourInfoNode[i])
# print("角度信息")
# print(angleInfo)


def outputcontourinfo():
    i = 0
    for a in contourInfo:
        print("----------------%d" % i)
        j = 0
        for b in a.contourInfoNode:
            print("%d" % j, b)
            j += 1
        i += 1


print("output contourinfo information")
outputcontourinfo()
# i=1
# for contour in contours:
#     blacktemp=blacktemp1.copy()
#     color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
#     color=(0,125,125)
#     cv2.drawContours(blacktemp, contours,i-1,color,5 )
#     plt.subplot(2,(len(contours)+1)/2,i)
#     plt.imshow(blacktemp)
#     plt.title('contour%d'%(i-1))
#     plt.xticks([]),plt.yticks([])
#     i+=1
# plt.show()
print("***********************")


class contourStitchInfoNode:
    def __init__(self, contourpos, begin, end, vertexNum):
        self.contourpos = contourpos
        self.begin = begin
        self.end = end
        self.vertexNum = vertexNum

    def __str__(self):
        return '("contourpos"%d"begin"%d"end"%d"vertexnum"%d)' % (self.contourpos, self.begin, self.end, self.vertexNum)


class contourStitchInfoBond:
    def __init__(self, vertexf, sidefb, sidefn, angleb, anglen, vertexs, sidesb, sidesn, direction):
        self.vertexf = vertexf
        self.sidefb = sidefb
        self.sidefn = sidefn
        self.angleb = angleb
        self.anglen = anglen
        self.vertexs = vertexs
        self.sidesb = sidesb
        self.sidesn = sidesn
        self.direction = direction


class contourStitchInfo:
    def __init__(self, contoursitchlist, vertexNum, contourStitched):
        self.contourstitchlist = contoursitchlist
        self.vertexNum = vertexNum
        self.contourStitched = contourStitched


class contourStitchPosition:
    def __init__(self, contour1, contour2, index1, index2):
        self.contour1 = contour1
        self.contour2 = contour2
        self.index1 = index1
        self.index2 = index2


class contourMatchVetex:
    def __init__(self, contourstitchpos, vertexpos):
        self.contourstitchpos = contourstitchpos
        self.vertexpos = vertexpos

    def __str__(self):
        return '("轮廓序号"%d"顶点序号"%d)' % (self.contourstitchpos, self.vertexpos)


class contourMatchNode:
    def __init__(self, fixbegin, matchbegin, fixend, matchend, matchDegree, errorDegree, isAnglematch, isValid):
        self.fixbegin = fixbegin
        self.matchbegin = matchbegin
        self.fixend = fixend
        self.matchend = matchend
        self.matchDegree = matchDegree
        self.errorDegree = errorDegree
        self.isAnglematch = isAnglematch
        self.isValid = isValid

    def __str__(self):
        return '("x起始点"(%d,%d)"y起点"(%d,%d)"x终点"(%d,%d)"y终点"(%d,%d)"匹配程度"%d,"error"%d,"角度匹配否"%d"有效否"%d)' % (self.fixbegin.stitchpos, self.fixbegin.vertexpos, self.matchbegin.stitchpos, self.matchbegin.vertexpos, self.fixend.stitchpos, self.fixend.vertexpos, self.matchend.stitchpos, self.matchend.vertexpos, self.matchDegree, self.errorDegree, self.isAnglematch, self.isValid)


class contourStitchIndex:
    def __init__(self, stitchpos, vertexpos):
        self.stitchpos = stitchpos
        self.vertexpos = vertexpos


def contourstichnext(stitchIndex, contourStitchList):
    '''找到拼合的下一个点\n
    返回值(fixpos,xfix, fixy, vertexfix)'''
    stitchpos = stitchIndex.stitchpos
    vertexpos = stitchIndex.vertexpos
    vertexposend = contourStitchList[stitchpos].end
    vertexnum = contourStitchList[stitchpos].vertexNum
    # 是否到了下一个轮廓
    if(vertexpos == vertexposend):
        stitchpos += 1
        if(stitchpos >= len(contourStitchList)):
            stitchpos = 0
        vertexpos = contourStitchList[stitchpos].begin
    else:
        vertexpos = (vertexpos+1) % vertexnum
    stitchIndex.stitchpos = stitchpos
    stitchIndex.vertexpos = vertexpos
    contourpos = contourStitchList[stitchpos].contourpos
    vertex = contourInfo[contourpos].contourInfoNode[vertexpos]
    return vertex


def contourstichpre(stitchIndex, contourStitchList):
    '''找到拼合的上一个点\n
    返回值(stitchIndex,vertex)'''
    stitchpos = stitchIndex.stitchpos
    vertexpos = stitchIndex.vertexpos
    vertexposbegin = contourStitchList[stitchpos].begin
    vertexnum = contourStitchList[stitchpos].vertexNum
    # 是否到了下一个轮廓
    if(vertexpos == vertexposbegin):
        stitchpos -= 1
        if(stitchpos < 0):
            stitchpos = len(contourStitchList)-1
        vertexpos = contourStitchList[stitchpos].end
    else:
        vertexpos = (vertexpos-1) % vertexnum
    stitchIndex.stitchpos = stitchpos
    stitchIndex.vertexpos = vertexpos
    contourpos = contourStitchList[stitchpos].contourpos
    vertex = contourInfo[contourpos].contourInfoNode[vertexpos]
    return vertex


def vertexposless(contourstitchnode, xpos, ypos):
    'suppose xpos and ypos in range'
    posbegin = contourstitchnode.begin
    if((xpos >= posbegin and ypos >= posbegin)or(xpos < posbegin and ypos < posbegin)):
        return xpos < ypos
    elif(xpos >= posbegin):
        return True
    else:
        return False


class contourMatchAllInfo:
    def __init__(self, contourmatchpairlist, alonefixpos, alonematchpos):
        self.contourmatchpairlist = contourmatchpairlist
        self.alonefixpos = alonefixpos
        self.alonematchpos = alonematchpos


def constructMatchPair(contourinfo, fixcontourstitchinfo, matchcontourstitchinfo):
    contourmatchpairlist = list()
    fixnum = fixcontourstitchinfo.vertexNum
    matchnum = matchcontourstitchinfo.vertexNum
    fixcontourstitchlist = fixcontourstitchinfo.contourstitchlist
    matchcontourstitchlist = matchcontourstitchinfo.contourstitchlist
    fixstitchpos1 = 0
    fixvertexpos1 = fixcontourstitchlist[fixstitchpos1].begin
    fixstitchindex1 = contourStitchIndex(fixstitchpos1, fixvertexpos1)
    matchmax = min(matchnum, fixnum)
    for j in range(fixnum):
        fixvertex1 = contourstichnext(
            fixstitchindex1, fixcontourstitchlist)
        matchstitchpos1 = 0
        matchvertexpos1 = matchcontourstitchlist[matchstitchpos1].begin
        matchstitchindex1 = contourStitchIndex(
            matchstitchpos1, matchvertexpos1)
        for k in range(matchnum):
            matchvertex1 = contourstichpre(
                matchstitchindex1, matchcontourstitchlist)
            fixvertex = copy.deepcopy(fixvertex1)
            fixstitchindex = copy.deepcopy(fixstitchindex1)
            matchvertex = copy.deepcopy(matchvertex1)
            matchstitchindex = copy.deepcopy(matchstitchindex1)
            matchDegree = 0
            errorDegree = 0
            isAnglematch = False
            matchFlag = True
            fixvertextemp= contourstichpre(
                copy.deepcopy(fixstitchindex), fixcontourstitchlist)
            matchvertextemp = contourstichnext(
                copy.deepcopy(matchstitchindex), matchcontourstitchlist)
            if((fixvertex.angle+matchvertex.angle) > 370):
                continue
            if(abs(360-fixvertex.angle-matchvertex.angle) <= 10):
                isAnglematch = True
                if(abs(fixvertex.lasside-matchvertex.nexside) < 10):
                    continue
                if(fixvertex.lasside < matchvertex.nexside and fixvertextemp.angle > 190):
                    continue
                if(fixvertex.lasside > matchvertex.nexside and matchvertextemp.angle > 190):
                    continue
                # matchDegree += 1
            while(matchDegree < 2*matchmax):
                if(abs(fixvertex.nexside-matchvertex.lasside) > 10):
                    if(fixvertex.nexside > matchvertex.lasside):
                        matchvertextemp = contourstichpre(
                            copy.deepcopy(matchstitchindex), matchcontourstitchlist)
                        matchFlag = (matchvertextemp.angle < 190)
                    else:
                        fixvertextemp = contourstichnext(
                            copy.deepcopy(fixstitchindex), fixcontourstitchlist)
                        matchFlag = (fixvertextemp.angle < 190)
                    break
                elif(abs(fixvertex.nexside-matchvertex.lasside) > 5):
                    errorDegree += 1
                matchDegree += 1
                fixvertex = contourstichnext(
                    fixstitchindex, fixcontourstitchlist)
                matchvertex = contourstichpre(
                    matchstitchindex, matchcontourstitchlist)
                if(abs(360-fixvertex.angle-matchvertex.angle) > 10):
                    if((fixvertex.angle+matchvertex.angle) > 370):
                        matchFlag = False
                    break
                elif(abs(360-fixvertex.angle-matchvertex.angle) > 5):
                    errorDegree += 1
                matchDegree += 1
            if(matchDegree >= 1 and matchFlag):
                matchnodeA = contourMatchNode(copy.deepcopy(fixstitchindex1), copy.deepcopy(matchstitchindex1), copy.deepcopy(fixstitchindex),
                                              copy.deepcopy(matchstitchindex), matchDegree, errorDegree, isAnglematch, True)
                contourmatchpairlist.append(matchnodeA)
    return contourmatchpairlist


def constructMatchAll(contourinfo, contourstitchall):
    contourmatchallinfo = list()
    for i in range(len(contourstitchall)):
        for j in range(i+1, len(contourstitchall)):
            contourmatchpairlist = constructMatchPair(
                contourinfo, contourstitchall[i], contourstitchall[j])
            if(contourmatchpairlist):
                contourmatchallinfo.append(
                    contourMatchAllInfo(contourmatchpairlist, i, j))
    return contourmatchallinfo


def calculateVertexnum(contourstitchinfolist):
    vertexnum = 0
    for contourstitchnode in contourstitchinfolist:
        if(contourstitchnode.end < contourstitchnode.begin):
            vertexnum += (contourstitchnode.vertexNum -
                          contourstitchnode.begin+contourstitchnode.end+1)
        else:
            vertexnum += (contourstitchnode.end-contourstitchnode.begin+1)
    return vertexnum


def changeStitchList(contourstitchlist, beginstitchindex, endstitchindex):
    beginstitchpos=beginstitchindex.stitchpos
    beginvertexpos=beginstitchindex.vertexpos
    endstitchpos=endstitchindex.stitchpos   
    endvertexpos=endstitchindex.vertexpos
    if(endstitchpos == beginstitchpos and vertexposless(contourstitchlist[beginstitchpos], beginvertexpos, endvertexpos)):
        temp =copy.deepcopy(contourstitchlist[beginstitchpos]) 
        temp.begin=endvertexpos
        contourstitchlist.insert(beginstitchpos+1, temp)
        contourstitchlist[beginstitchpos].end = beginvertexpos
        return beginstitchpos+1
    elif(endstitchpos > beginstitchpos):
        contourstitchlist[beginstitchpos].end = beginvertexpos
        contourstitchlist[endstitchpos].begin = endvertexpos
        for j in range(beginstitchpos+1, endstitchpos):
            contourstitchlist.pop(beginstitchpos+1)
        return beginstitchpos+1
    else:
        contourstitchlist[beginstitchpos].end = beginvertexpos
        contourstitchlist[endstitchpos].begin = endvertexpos
        for j in range(beginstitchpos+1, len(contourstitchlist)):
            contourstitchlist.pop()
        for j in range(endstitchpos):
            contourstitchlist.pop(0)
        return len(contourstitchlist)


contourinfokeep = copy.deepcopy(contourInfo)
bottom = 0


def backtrace(allcontourstitch):
    if(len(allcontourstitch) == 1):
        # bottom+=1
        contourlistfinal = allcontourstitch[0].contourstitchlist
        vertexnum = calculateVertexnum(contourlistfinal)
        vertexpos=contourlistfinal[0].begin
        stitchindex=contourStitchIndex(0,vertexpos)
        validAngleNum = 0
        for j in range(vertexnum):
            vertex = contourstichnext(
                stitchindex, contourlistfinal)
            if(vertex.angle < 170 or vertex.angle > 190):
                validAngleNum += 1
        if(validAngleNum < 100):
            contourpiclist = list()
            for j in allcontourstitch[0].contourStitched:
                contournode = contourInfo[j].contourInfoNode
                vertexnum = len(contournode)
                contourpic = np.empty(shape=(vertexnum, 1, 2), dtype=int)
                for k in range(vertexnum):
                    contourpic[k][0] = contournode[k].prepoint
                contourpiclist.append(contourpic)

                # paint the single pic
            blacktemp = black.copy()
            cv2.drawContours(blacktemp, contourpiclist, -1, (0, 255, 0), 8)
            cv2.namedWindow("BGR", 0)
            cv2.resizeWindow("BGR", 500, 500)
            cv2.imshow('BGR', blacktemp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return True
    contourmatchinfo = constructMatchAll(contourInfo, allcontourstitch)
    while True:
        contourstitchallbefore = copy.deepcopy(allcontourstitch)
# 寻找最大值
        alonefixpos=0
        alonematchpos=0
        matchMaxNode = contourMatchNode(contourStitchIndex(-1,-1), contourStitchIndex(0,0), contourStitchIndex(0,0),contourStitchIndex(0,0),0, 0, False, False)
        for contourmatchpairinfo in contourmatchinfo:
            contourmatchlist = contourmatchpairinfo.contourmatchpairlist
            for contourmatchnode in contourmatchlist:
                if(contourmatchnode.isValid and (contourmatchnode.matchDegree > matchMaxNode.matchDegree or (contourmatchnode.matchDegree == matchMaxNode.matchDegree and contourmatchnode.errorDegree < matchMaxNode.errorDegree))):
                    matchMaxNode = contourmatchnode
                    alonefixpos=contourmatchpairinfo.alonefixpos
                    alonematchpos=contourmatchpairinfo.alonematchpos
        if(matchMaxNode.fixbegin.stitchpos==-1):
            break
        matchMaxNode.isValid = False
# 更新matchstitchinfo
        fixcontourstitchinfo=allcontourstitch[alonefixpos]
        fixcontourstitchlist = fixcontourstitchinfo.contourstitchlist
        fixcontourstitched=fixcontourstitchinfo.contourStitched
        fixvertexnum=fixcontourstitchinfo.vertexNum
        fixbegin = matchMaxNode.fixbegin
        fixbeginstitchpos = fixbegin.stitchpos
        fixbeginvertexpos = fixbegin.vertexpos
        fixbegincontourpos = fixcontourstitchlist[fixbeginstitchpos].contourpos
        fixbeginvertex = contourInfo[fixbegincontourpos].contourInfoNode[fixbeginvertexpos]
        fixend = matchMaxNode.fixend
        fixendstitchpos = fixend.stitchpos
        fixendvertexpos = fixend.vertexpos
        fixendcontourpos = fixcontourstitchlist[fixendstitchpos].contourpos
        fixendvertex = contourInfo[fixendcontourpos].contourInfoNode[fixendvertexpos]

        # i = 0
        # for contourstitchnode in contourstitchlist:
        #     if(contourstitchnode.index == fixbegin.contourindex):
        #         beginstitchpos = i
        #     if(contourstitchnode.index == fixend.contourindex):
        #         endstitchpos = i
        #         i += 1
        matchcontourstitchinfo=allcontourstitch[alonematchpos]
        matchcontourstitchlist = matchcontourstitchinfo.contourstitchlist
        matchcontourstitched=matchcontourstitchinfo.contourStitched
        matchvertexnum=matchcontourstitchinfo.vertexNum
        matchbegin = matchMaxNode.matchbegin
        matchbeginstitchpos = matchbegin.stitchpos
        matchbeginvertexpos = matchbegin.vertexpos
        matchbegincontourpos = matchcontourstitchlist[matchbeginstitchpos].contourpos
        matchbeginvertex = contourInfo[matchbegincontourpos].contourInfoNode[matchbeginvertexpos]
        matchend = matchMaxNode.matchend
        matchendstitchpos = matchend.stitchpos
        matchendvertexpos = matchend.vertexpos
        matchendcontourpos = matchcontourstitchlist[matchendstitchpos].contourpos
        matchendvertex = contourInfo[matchendcontourpos].contourInfoNode[matchendvertexpos]

# #paint contourstitch 
#         if(step==1):
#             vertexnum=fixcontourstitchinfo.vertexNum
#             contourallpic=np.empty(shape=(vertexnum,1,2),dtype=np.int)
#             vertexpos=fixcontourstitchlist[0].begin
#             fixstitchindex=contourStitchIndex(0,vertexpos)
#             for j in range(vertexnum):
#                 vertex=contourstichnext(fixstitchindex,fixcontourstitchlist)
#                 contourallpic[j][0]=vertex.prepoint
    
#             blacktemp=black.copy()
#             cv2.drawContours(blacktemp,[contourallpic],0,(0,255,0),3)
#             cv2.namedWindow("gray",0)
#             cv2.imshow('gray',blacktemp)
#             cv2.resizeWindow("gray",100,100)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()  

# #paint contourstitch 
#             vertexnum=matchcontourstitchinfo.vertexNum
#             contourallpic=np.empty(shape=(vertexnum,1,2),dtype=np.int)
#             vertexpos=matchcontourstitchlist[0].begin
#             matchstitchindex=contourStitchIndex(0,vertexpos)
#             for j in range(vertexnum):
#                 vertex=contourstichnext(matchstitchindex,matchcontourstitchlist)
#                 contourallpic[j][0]=vertex.prepoint
    
#             blacktemp=black.copy()
#             cv2.drawContours(blacktemp,[contourallpic],0,(0,255,0),3)
#             cv2.namedWindow("gray",0)
#             cv2.imshow('gray',blacktemp)
#             cv2.resizeWindow("gray",100,100)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()  

        fixstitchindex = copy.deepcopy(matchMaxNode.fixbegin)
        fixvertex = fixbeginvertex
        matchstitchindex = copy.deepcopy(matchMaxNode.matchbegin)
        matchvertex = matchbeginvertex

# save prepoint info
        contourinfokeepfix=dict()
        contourinfokeepmatch=dict()
        for j in fixcontourstitchinfo.contourStitched:
            contourinfokeepfix[j]=copy.deepcopy(contourInfo[j].contourInfoNode)
        for j in matchcontourstitchinfo.contourStitched:
            contourinfokeepmatch[j]=copy.deepcopy(contourInfo[j].contourInfoNode)

        # 对合并后的角度进行剔除或修改
        if(not matchMaxNode.isAnglematch):
            # # 角度不互补，边长相等
            # 更新contourstitchinfo信息
            angleafter = matchvertex.angle+fixvertex.angle
            fixvertex = contourstichpre(
                fixstitchindex, fixcontourstitchlist)
            matchBonds = contourStitchInfoBond(matchvertex, matchvertex.lasside, fixvertex.nexside,
                                               matchvertex.angle, angleafter, fixvertex, fixvertex.nexside, fixvertex.nexside, False)

        else:
            # 角度互补，边长不相等
            # 固定的比较长
            if(matchvertex.nexside < fixvertex.lasside):
                # 更新contourstitchinfo信息
                laslength = fixvertex.lasside-matchvertex.nexside
                matchvertex = contourstichnext(
                    matchstitchindex, matchcontourstitchlist)
                angleafter = matchvertex.angle+180
                fixvertex = contourstichpre(
                    fixstitchindex, fixcontourstitchlist)
                matchBonds = contourStitchInfoBond(matchvertex, matchvertex.lasside, laslength,
                                                   matchvertex.angle, angleafter, fixvertex, fixvertex.nexside, laslength, False)

            else:
                # # 匹配的比较长
                # 更新contourstitchinfo信息
                nexlength = matchvertex.nexside-fixvertex.lasside
                fixvertex = contourstichpre(
                    fixstitchindex, fixcontourstitchlist)
                angleafter = fixvertex.angle+180
                matchvertex = contourstichnext(
                    matchstitchindex, matchcontourstitchlist)
                matchBonds = contourStitchInfoBond(
                    fixvertex, fixvertex.nexside, nexlength, fixvertex.angle, angleafter, matchvertex, matchvertex.lasside, nexlength, True)
        fixbeginstitchindex = fixstitchindex
        matchbeginstitchindex = matchstitchindex

        fixstitchindex = copy.deepcopy(matchMaxNode.fixend)
        fixvertex = fixendvertex
        matchstitchindex = copy.deepcopy(matchMaxNode.matchend)
        matchvertex = matchendvertex

        matchdegree = matchMaxNode.matchDegree
        # if(matchMaxNode.isAnglematch):
        #     matchdegree -= 1
        # 对合并后的角度进行剔除或修改
        if((matchdegree % 2) == 1):
            # 角度不互补，边长相等
            angleafter = matchvertex.angle+fixvertex.angle
            fixvertex = contourstichnext(
                fixstitchindex, fixcontourstitchlist)
            matchBondr = contourStitchInfoBond(matchvertex, matchvertex.nexside, fixvertex.lasside,
                                               matchvertex.angle, angleafter, fixvertex, fixvertex.lasside, fixvertex.lasside, True)

        else:
            # 角度互补，边长不相等
            if(matchvertex.lasside < fixvertex.nexside):
                # 固定的长度长
                # 计算分离点的角度，边长信息
                nexlength = fixvertex.nexside-matchvertex.lasside
                matchvertex = contourstichpre(
                    matchstitchindex, matchcontourstitchlist)
                angleafter = matchvertex.angle+180
                fixvertex = contourstichnext(
                    fixstitchindex, fixcontourstitchlist)
                matchBondr = contourStitchInfoBond(matchvertex, matchvertex.nexside, nexlength,
                                                   matchvertex.angle, angleafter, fixvertex, fixvertex.lasside, nexlength, True)
            else:
                # 匹配的长度长
                # 计算分离点的角度，边长信息
                laslength = matchvertex.lasside-fixvertex.nexside
                fixvertex = contourstichnext(
                    fixstitchindex, fixcontourstitchlist)
                angleafter = fixvertex.angle+180
                matchvertex = contourstichpre(
                    matchstitchindex, matchcontourstitchlist)
                matchBondr = contourStitchInfoBond(
                    fixvertex, fixvertex.lasside, laslength, fixvertex.angle, angleafter, matchvertex, matchvertex.nexside, laslength, False)

        fixendstitchindex = fixstitchindex
        matchendstitchindex = matchstitchindex
# remove matchcontour
        allcontourstitch.remove(matchcontourstitchinfo)
# change contourstitch info
        matchcontourstitchlistkeep=copy.deepcopy(matchcontourstitchlist)
        fixcontourstitchlistkeep=copy.deepcopy(fixcontourstitchlist)
        fixstitchpos=changeStitchList(fixcontourstitchlist,fixbeginstitchindex,fixendstitchindex)
        matchstitchpos=changeStitchList(matchcontourstitchlist,matchendstitchindex,matchbeginstitchindex)
        pos=matchstitchpos
        nodenum=len(matchcontourstitchlist)
        for i in range(nodenum):
            pos=pos%nodenum
            fixcontourstitchlist.insert(fixstitchpos,matchcontourstitchlist[pos])
            fixstitchpos+=1
            pos+=1
 # change vertexnum info
        fixcontourstitchinfo.vertexNum = calculateVertexnum(fixcontourstitchlist)

# change contourstitched info
        fixkeep=copy.deepcopy(fixcontourstitched)
        fixcontourstitched+=matchcontourstitched

# change vertex info
        if(matchBondr.direction):
            matchBondr.vertexf.nexside = matchBondr.sidefn
            matchBondr.vertexf.angle = matchBondr.anglen
            matchBondr.vertexs.lasside = matchBondr.sidesn
        else:
            matchBondr.vertexf.lasside = matchBondr.sidefn
            matchBondr.vertexf.angle = matchBondr.anglen
            matchBondr.vertexs.nexside = matchBondr.sidesn
        if(matchBonds.direction):
            matchBonds.vertexf.nexside = matchBonds.sidefn
            matchBonds.vertexf.angle = matchBonds.anglen
            matchBonds.vertexs.lasside = matchBonds.sidesn
        else:
            matchBonds.vertexf.lasside = matchBonds.sidefn
            matchBonds.vertexf.angle = matchBonds.anglen
            matchBonds.vertexs.nexside = matchBonds.sidesn

# rotate and shift
        fix = fixendvertex.prepoint-fixbeginvertex.prepoint
        match = matchendvertex.prepoint-matchbeginvertex.prepoint
        isReserve = ((fix[0]*match[1]-fix[1]*match[0]) > 0)
        fixlength = math.sqrt(fix[0]*fix[0]+fix[1]*fix[1])
        matchlength = math.sqrt(match[0]*match[0]+match[1]*match[1])
        if(fixlength == 0 or matchlength == 0):
            print("invalid value")
            contournode = contourInfo[4].contourInfoNode
            alone = np.empty(shape=(len(contournode), 1, 2), dtype=np.int)
            for j in range(len(contournode)):
                alone[j][0] = contournode[j].prepoint

            # paint the single pic
            blacktemp = black.copy()
            # cv2.drawContours(blacktemp,[contourpicbefore],0,(0,255,0),3)
            cv2.drawContours(blacktemp, [alone], 0, (0, 0, 255), 3)
            plt.imshow(blacktemp)
        cosrotateangle = ((fix[0]*match[0]) +
                          (fix[1]*match[1]))/(fixlength*matchlength)
        if(cosrotateangle < -1):
            cosrotateangle = -1
        if(cosrotateangle > 1):
            cosrotateangle = 1
        if(cosrotateangle > 1 or cosrotateangle < -1):
            print("angle out of domain")
        rotateangle = math.acos(cosrotateangle)
        if(isReserve):
            rotateangle = -rotateangle
        rotatefactor = np.array([[math.cos(rotateangle), -math.sin(rotateangle)], [
                                math.sin(rotateangle), math.cos(rotateangle)]])
        coordinatefix = matchbeginvertex.prepoint
        shift = fixbeginvertex.prepoint-matchbeginvertex.prepoint
        # contourarray=np.empty(shape=(len(contournode),1,2),dtype=int)
        for contourpos in matchcontourstitched:
            contournode=contourInfo[contourpos].contourInfoNode
            for j in range(len(contournode)):
                coordinaterotate = contournode[j].prepoint
                if(id(coordinatefix)!=id(coordinaterotate)):
                    coordinatedif = coordinaterotate-coordinatefix
                    coordinateafterrotate = np.dot(
                        rotatefactor, coordinatedif)+coordinatefix
                    coordinateaftershift = coordinateafterrotate+shift
                    contournode[j].prepoint = coordinateaftershift.astype(np.int)
                # else:
                    # print("have a fix")
        coordinatefix+= shift   
    # # paint the contour
    #     if(step==1):
    #         matchpic=np.empty(shape=(matchvertexnum,1,2),dtype=np.int)
    #         vertexpos=matchcontourstitchlistkeep[0].begin
    #         matchstitchindex=contourStitchIndex(0,vertexpos)
    #         for j in range(matchvertexnum):
    #             matchvertex = contourstichpre(
    #                     matchstitchindex, matchcontourstitchlistkeep)
    #             matchpic[j][0]=matchvertex.prepoint
    #         fixpic=np.empty(shape=(fixvertexnum,1,2),dtype=np.int)
    #         vertexpos=fixcontourstitchlistkeep[0].begin
    #         fixstitchindex=contourStitchIndex(0,vertexpos)
    #         for j in range(fixvertexnum):
    #             fixvertex = contourstichnext(
    #                     fixstitchindex, fixcontourstitchlistkeep)
    #             fixpic[j][0]=fixvertex.prepoint
    #         blacktemp=black.copy()
    #         cv2.drawContours(blacktemp,[fixpic],0,(0,255,0),3)
    #         cv2.drawContours(blacktemp, [matchpic], 0, (0, 0, 255), 3)
    #         cv2.namedWindow("gray",0)
    #         cv2.imshow('gray',blacktemp)
    #         cv2.resizeWindow("gray",100,100)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    # for debug
        if(step==1):
            print("-----------------------go to anothor")
            print("contourmatchlist----:")
            for a in contourmatchinfo:
                print("alonefixpos:%d,alonematchpos:%d"%(a.alonefixpos,a.alonematchpos))
                for contourmatchpair in a.contourmatchpairlist:
                    print(contourmatchpair)
            print("matchmaxnode----------:")
            print(matchMaxNode)
            print("stitchinfo------:")
            for a in fixcontourstitchlist:
                print(a)
            print("contourstitched----:")
            print(fixcontourstitchinfo.contourStitched)
            print("contourstitchall-----:")
            for contourstitch in allcontourstitch:
                print(contourstitch.contourStitched)
            # output vertex info
            print("vertexinfo--------:")
# calculate the center
        vertexpos=fixcontourstitchlist[0].begin
        fixstitchindex=contourStitchIndex(0,vertexpos)
        coordinatesum = np.array([0, 0], dtype=int)
        for j in range(fixcontourstitchinfo.vertexNum):
            vertex = contourstichnext(
                fixstitchindex, fixcontourstitchlist)
            coordinatesum += vertex.prepoint
            # print(vertex)
        coordinate_average = np.floor_divide(
            coordinatesum, fixcontourstitchinfo.vertexNum)
        coordinate_average.astype(np.int)
# move the stitch to center
        coordinatecenter = np.array([1000, 1000], dtype=int)
        coordinatecenterdif = coordinatecenter-coordinate_average
        for contourpos in fixcontourstitched:
            for vertex in contourInfo[contourpos].contourInfoNode:
                vertex.prepoint += coordinatecenterdif
    # paint the contour
        if(step==1):
            matchpic=np.empty(shape=(matchvertexnum,1,2),dtype=np.int)
            vertexpos=matchcontourstitchlistkeep[0].begin
            matchstitchindex=contourStitchIndex(0,vertexpos)
            for j in range(matchvertexnum):
                matchvertex = contourstichpre(
                        matchstitchindex, matchcontourstitchlistkeep)
                matchpic[j][0]=matchvertex.prepoint
            fixpic=np.empty(shape=(fixvertexnum,1,2),dtype=np.int)
            vertexpos=fixcontourstitchlistkeep[0].begin
            fixstitchindex=contourStitchIndex(0,vertexpos)
            for j in range(fixvertexnum):
                fixvertex = contourstichnext(
                        fixstitchindex, fixcontourstitchlistkeep)
                fixpic[j][0]=fixvertex.prepoint

            blacktemp=black.copy()
            cv2.drawContours(blacktemp,[fixpic],0,(0,255,0),3)
            cv2.drawContours(blacktemp, [matchpic], 0, (0, 0, 255), 3)
            cv2.namedWindow("gray",0)
            cv2.imshow('gray',blacktemp)
            cv2.resizeWindow("gray",100,100)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # #paint contourstitch 
    #         vertexnum=fixcontourstitchinfo.vertexNum
    #         contourallpic=np.empty(shape=(vertexnum,1,2),dtype=np.int)
    #         vertexpos=fixcontourstitchlist[0].begin
    #         fixstitchindex=contourStitchIndex(0,vertexpos)
    #         for j in range(vertexnum):
    #             vertex=contourstichnext(fixstitchindex,fixcontourstitchlist)
    #             contourallpic[j][0]=vertex.prepoint
    
    #         blacktemp=black.copy()
    #         cv2.drawContours(blacktemp,[contourallpic],0,(0,255,0),3)
    #         cv2.namedWindow("gray",0)
    #         cv2.imshow('gray',blacktemp)
    #         cv2.resizeWindow("gray",100,100)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()       
  
# backtrace
        backtrace(allcontourstitch)
# back to before
        if(step==1):
            print("-----------------------back to before")
# change back state
        if(matchBondr.direction):
            matchBondr.vertexf.nexside = matchBondr.sidefb
            matchBondr.vertexf.angle = matchBondr.angleb
            matchBondr.vertexs.lasside = matchBondr.sidesb
        else:
            matchBondr.vertexf.lasside = matchBondr.sidefb
            matchBondr.vertexf.angle = matchBondr.angleb
            matchBondr.vertexs.nexside = matchBondr.sidesb
        if(matchBonds.direction):
            matchBonds.vertexf.nexside = matchBonds.sidefb
            matchBonds.vertexf.angle = matchBonds.angleb
            matchBonds.vertexs.lasside = matchBonds.sidesb
        else:
            matchBonds.vertexf.lasside = matchBonds.sidefb
            matchBonds.vertexf.angle = matchBonds.angleb
            matchBonds.vertexs.nexside = matchBonds.sidesb
        allcontourstitch = contourstitchallbefore
        fixcontourstitchinfo=allcontourstitch[alonefixpos]
        for j in fixcontourstitchinfo.contourStitched:
            contourInfo[j].contourInfoNode=contourinfokeepfix[j]
        matchcontourstitchinfo=allcontourstitch[alonematchpos]
        for j in matchcontourstitchinfo.contourStitched:
            contourInfo[j].contourInfoNode=contourinfokeepmatch[j]

    return False


def testbacktrace():
# step by step or not
    x=input('choose to step by step or not, 0 to step ,1 to not')
    global step
    step=int(x)
#init 
    allcontourstitch = list()
    contourstitched = list()
    contournum = len(contourInfo)
    for i in range(contournum):
        vertexnum = contourInfo[i].vertexNum
        contourstitchnode = contourStitchInfoNode(
            i, 0, vertexnum-1, vertexnum)
        contourstitchnodelist=list()
        contourstitchnodelist.append(contourstitchnode)
        contourstitched = [i]
        alonecontourstitch = contourStitchInfo(
            contourstitchnodelist, vertexnum, contourstitched)
        allcontourstitch.append(alonecontourstitch)
    if(backtrace(allcontourstitch)):
        print("***************")
    else:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


print("test testbacktrace() function")
testbacktrace()
# 画轮廓
# cv2.drawContours(black, contours, -1, (0, 255, 0), 3)

# # 用subplot显
# # ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# # ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# # ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# # ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# # titles = ['ORIGNAL Image','BGR2GRAY Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# # images = [orignal,img, thresh1, thresh2, thresh3, thresh4, thresh5]
# # for i in range(7):
# #     plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
# #     plt.title(titles[i])
# #     plt.xticks([]),plt.yticks([])
# # plt.show()
# # white=cv2.imread('/home/hurly/Desktop/white.png')
# # img=cv2.add(img,white)

# # 单独显示
# # cv2.namedWindow('Image')
# cv2.imshow('BGR', black)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
