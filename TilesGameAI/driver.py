from math import sqrt
from collections import deque
from collections import Set
import time
import sys
import os
import copy
from copy import deepcopy
from operator import or_
from lib2to3.fixer_util import Number
import Queue
import resource


class Cmove:
    Up, Down, Left, Right = range(4)
    

class board:
    #class storing a board
 
    def isValid(self):
        if not self.tileset:
            return False
        return True;
 
    def __init__(self, tileset):
        self.tileset = tileset
        
    def getTileset(self):
        return self.tileset

    def editTileset(self, tileset, places):
        ind0 = tileset.index(0)
        indEl = ind0+places
        
        n = sqrt(len(tileset))
        
        if (indEl<0):
            return list()
        if (indEl>=len(tileset)):
            return list()
        if( ind0%n == n-1 ):
            if (indEl%n == 0):
                return list()
        if( ind0%n == 0):
            if( indEl%n == n-1):
                return list()
        
        newTileset = tileset[:]
        newTileset[ind0], newTileset[indEl] = newTileset[indEl], newTileset[ind0]
        
        return newTileset
 
    def movement(self, move):
  
        if(move == Cmove.Up):
            n = int(sqrt(len(self.tileset)))
            newTileset = self.editTileset(self.tileset,-n)
        elif(move == Cmove.Down):
            n = int(sqrt(len(self.tileset)))
            newTileset = self.editTileset(self.tileset,n)
        elif(move == Cmove.Left):
            newTileset = self.editTileset(self.tileset,-1)
        else:
            newTileset = self.editTileset(self.tileset,1)
   
        return board(newTileset)
    
    @staticmethod
    def getNumberDistance(n, index, element):
        difference = abs(index-element)
        distance = 0
        
        while(difference>=n):
            difference = difference - n
            distance = distance+1
            
        distance = distance + difference
        
        return distance
        
    
    def manhattanDistance(self):
        n = int(sqrt(len(self.tileset)))
        sumDistance = 0
        
        for i in self.tileset:
            distance = self.getNumberDistance(n, self.tileset.index(i), i)
            sumDistance = sumDistance + distance
        
        return sumDistance
        
        
    
class searchNode:
    def __init__(self, board, moveList):
        self.board = board
        self.moves = moveList
    
    def isValid(self):
        return self.getBoard().isValid()
    
    def getBoard(self):
        return self.board
    
    def getMoveList(self):
        return self.moves
    
    def getDepth(self):
        return len(self.moves)
    
    def getNodeFullManhatanCost(self):
        cost = self.getBoard().manhattanDistance()
        cost = cost + len(self.moves)
        return cost
        
    def createGetChildNode(self, move, newBoard):
        if(move==Cmove.Up):
            newMoveList = deepcopy(self.getMoveList())
            newMoveList.append("Up")
            return searchNode(newBoard, newMoveList)
        elif(move==Cmove.Down):
            newMoveList = deepcopy(self.getMoveList())
            newMoveList.append("Down")
            return searchNode(newBoard, newMoveList)
        elif(move==Cmove.Left):
            newMoveList = deepcopy(self.getMoveList())
            newMoveList.append("Left")
            return searchNode(newBoard, newMoveList)
        else:
            newMoveList = deepcopy(self.getMoveList())
            newMoveList.append("Right")
            return searchNode(newBoard, newMoveList)

class solver:
        
    @staticmethod
    def bfs(rootNode, final):
        #Breadth-First Search
        
        global explored
        global fringe
        global maxFringe
        global maxDepth
        global maxRam

        
        maxDepth = 0
        maxFringe = 0
        explored = set([])
        maxRam = 1342.35
        currRam = 0.15
        
        fringe = deque([rootNode])
        fringeBoards = set()
        fringeBoards.add(tuple(rootNode.getBoard().getTileset()))
        
        while(len(fringe)!=0):
            
            if(len(fringe)>maxFringe):
                maxFringe = len(fringe)
            
            currentNode = fringe.popleft()
            fringeBoards.remove(tuple(currentNode.getBoard().getTileset()))
            
            if(currentNode.getDepth()>maxDepth):
                maxDepth = currentNode.getDepth()
            
            if(currentNode.getBoard().getTileset()==final):
                return currentNode
            
            explored.add(tuple(currentNode.getBoard().getTileset()))
            
            currBoard = currentNode.getBoard()
            newDepth = currentNode.getDepth()+1
            
            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Up, currentBoard.movement(Cmove.Up))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                    fringe.append(newNode)
                    fringeBoards.add(tuple(newNode.getBoard().getTileset()))


            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Down, currentBoard.movement(Cmove.Down))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                    fringe.append(newNode)
                    fringeBoards.add(tuple(newNode.getBoard().getTileset()))


            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Left, currentBoard.movement(Cmove.Left))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                    fringe.append(newNode)
                    fringeBoards.add(tuple(newNode.getBoard().getTileset()))


            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Right, currentBoard.movement(Cmove.Right))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                    fringe.append(newNode)
                    fringeBoards.add(tuple(newNode.getBoard().getTileset()))

            
            if(newDepth>maxDepth):
                maxDepth = newDepth
            
                
            currRam = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if(currRam>maxRam):
                maxRam = currRam

        
        print "No solution found"
        return searchNode()
    
    @staticmethod
    def dfs(rootNode, final, depthLimit):
        #Depth-First Search
        global explored
        global fringe
        global maxFringe
        global maxDepth
        global maxRam

        
        maxDepth = 0
        maxFringe = 0
        explored = set([])
        maxRam = 1342.35
        currRam = 0.15
        
        fringe = [rootNode]
        fringeBoards = set()
        fringeBoards.add(tuple(rootNode.getBoard().getTileset()))
        
        
        while(len(fringe)!=0):
            
            if(len(fringe)>maxFringe):
                maxFringe = len(fringe)
            
            currentNode = fringe.pop()
            fringeBoards.remove(tuple(currentNode.getBoard().getTileset()))
            
            if(currentNode.getDepth()>maxDepth):
                maxDepth = currentNode.getDepth()
            
            if(currentNode.getBoard().getTileset()==final):
                return currentNode
            
            explored.add(tuple(currentNode.getBoard().getTileset()))
            
            currBoard = currentNode.getBoard()
            newDepth = currentNode.getDepth()+1
            
            if((depthLimit==0) | (newDepth<=depthLimit)):
                
                
                currentBoard = deepcopy(currBoard)
                newNode=currentNode.createGetChildNode(Cmove.Right, currentBoard.movement(Cmove.Right))
                if(newNode.isValid()):
                    if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                        fringe.append(newNode)
                        fringeBoards.add(tuple(newNode.getBoard().getTileset()))

            
                currentBoard = deepcopy(currBoard)
                newNode = currentNode.createGetChildNode(Cmove.Left, currentBoard.movement(Cmove.Left))
                if(newNode.isValid()):
                    if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                        fringe.append(newNode)
                        fringeBoards.add(tuple(newNode.getBoard().getTileset()))


                currentBoard = deepcopy(currBoard)
                newNode = currentNode.createGetChildNode(Cmove.Down, currentBoard.movement(Cmove.Down))
                if(newNode.isValid()):
                    if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                        fringe.append(newNode)
                        fringeBoards.add(tuple(newNode.getBoard().getTileset()))


                currentBoard = deepcopy(currBoard)
                newNode = currentNode.createGetChildNode(Cmove.Up, currentBoard.movement(Cmove.Up))
                if(newNode.isValid()):
                    if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                        fringe.append(newNode)
                        fringeBoards.add(tuple(newNode.getBoard().getTileset()))


                if(newDepth>maxDepth):
                    maxDepth = newDepth
                
            currRam = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if(currRam>maxRam):
                maxRam = currRam

        
        print "No solution found"
        return
    
    @staticmethod
    def ast(rootNode, final):
        #A-Star Search
        
        global explored
        global fringe
        global maxFringe
        global maxDepth
        global maxRam

        
        maxDepth = 0
        maxFringe = 0
        explored = set([])
        maxRam = 1342.35
        currRam = 0.15
        
        fringe = Queue.PriorityQueue()
        fringe.put(rootNode, rootNode.getNodeFullManhatanCost())
        
        
        while(not fringe.empty()):
            if(fringe.qsize()>maxFringe):
                maxFringe = fringe.qsize()
            
            while True:
                currentNode = fringe.get()
                if(tuple(currentNode.getBoard().getTileset()) not in explored):
                    break
                
                '''
                In order to be able to "update" a priority on the queue, we simply ignore
                the nodes already in the fringe. Heap in Python also ignores it, just changes the corresponding entry
                to null (REMOVED) and when it is retrieved, the code retrieves the next item. I implemented a way to do 
                this with a priority queue, simply ignoring items in the fringe as we input new elements in it, and when 
                they are retrieved a second time to be processed, we ignore them. No item (board) is processed twice.
                '''
            
            if(currentNode.getDepth()>maxDepth):
                maxDepth = currentNode.getDepth()
            
            if(currentNode.getBoard().getTileset()==final):
                fringe = range(fringe.qsize())
                return currentNode
            
            explored.add(tuple(currentNode.getBoard().getTileset()))
            
            currBoard = currentNode.getBoard()
            newDepth = currentNode.getDepth()+1
            
            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Up, currentBoard.movement(Cmove.Up))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in explored):
                    fringe.put(newNode, newNode.getNodeFullManhatanCost() )


            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Down, currentBoard.movement(Cmove.Down))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in explored):
                    fringe.put(newNode, newNode.getNodeFullManhatanCost() )


            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Left, currentBoard.movement(Cmove.Left))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in explored):
                    fringe.put(newNode, newNode.getNodeFullManhatanCost() )


            currentBoard = deepcopy(currBoard)
            newNode = currentNode.createGetChildNode(Cmove.Right, currentBoard.movement(Cmove.Right))
            if(newNode.isValid()):
                if(tuple(newNode.getBoard().getTileset()) not in explored):
                    fringe.put(newNode, newNode.getNodeFullManhatanCost() )
            
            
            if(newDepth>maxDepth):
                maxDepth = newDepth
            
            
            currRam = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if(currRam>maxRam):
                maxRam = currRam
        
        print "No solution found"
        return searchNode()

    
    @staticmethod
    def ida(rootNode, final):
        #IDA-Star Search
        
        global explored
        global fringe
        global maxFringe
        global maxDepth
        global maxRam

        
        maxDepth = 0
        maxFringe = 0
        explored = set([])
        maxRam = 1342.35
        currRam = 0.15
        
        limitslist = [rootNode.getNodeFullManhatanCost()]
        
        while (limitslist):
            limitslist.sort()
            limit = limitslist[0]

            fringe = [rootNode]
            fringeBoards = set()
            fringeBoards.add(tuple(rootNode.getBoard().getTileset()))
        
        
            while(len(fringe)!=0):
                
                if(len(fringe)>maxFringe):
                    maxFringe = len(fringe)
            
                currentNode = fringe.pop()
                fringeBoards.remove(tuple(currentNode.getBoard().getTileset()))
            
                if(currentNode.getDepth()>maxDepth):
                    maxDepth = currentNode.getDepth()
            
                if(currentNode.getBoard().getTileset()==final):
                    return currentNode
            
                explored.add(tuple(currentNode.getBoard().getTileset()))
            
                currBoard = currentNode.getBoard()
                newDepth = currentNode.getDepth()+1                
                
                currentBoard = deepcopy(currBoard)
                newNode=currentNode.createGetChildNode(Cmove.Right, currentBoard.movement(Cmove.Right))
                if(newNode.isValid()):
                    if(newNode.getNodeFullManhatanCost() <= limit):
                        if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                            fringe.append(newNode)
                            fringeBoards.add(tuple(newNode.getBoard().getTileset()))
                    else:
                        limitslist.append(newNode.getNodeFullManhatanCost())

            
                currentBoard = deepcopy(currBoard)
                newNode = currentNode.createGetChildNode(Cmove.Left, currentBoard.movement(Cmove.Left))
                if(newNode.isValid()):
                    if(newNode.getNodeFullManhatanCost() <= limit):
                        if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                            fringe.append(newNode)
                            fringeBoards.add(tuple(newNode.getBoard().getTileset()))
                    else:
                        limitslist.append(newNode.getNodeFullManhatanCost())


                currentBoard = deepcopy(currBoard)
                newNode = currentNode.createGetChildNode(Cmove.Down, currentBoard.movement(Cmove.Down))
                if(newNode.isValid()):
                    if(newNode.getNodeFullManhatanCost() <= limit):
                        if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                            fringe.append(newNode)
                            fringeBoards.add(tuple(newNode.getBoard().getTileset()))
                    else:
                        limitslist.append(newNode.getNodeFullManhatanCost())


                currentBoard = deepcopy(currBoard)
                newNode = currentNode.createGetChildNode(Cmove.Up, currentBoard.movement(Cmove.Up))
                if(newNode.isValid()):
                    if(newNode.getNodeFullManhatanCost() <= limit):
                        if(tuple(newNode.getBoard().getTileset()) not in (explored | fringeBoards)):
                            fringe.append(newNode)
                            fringeBoards.add(tuple(newNode.getBoard().getTileset()))
                    else:
                        limitslist.append(newNode.getNodeFullManhatanCost())


                if(newDepth>maxDepth):
                    maxDepth = newDepth
                
            currRam = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if(currRam>maxRam):
                maxRam = currRam
                
        
        print "No solution found"
        return
    
    
def finalizer(solution):
    resultString = "";
    
    global maxRam
    
    totalTime = time.time() - PrStartTime;
    
    maxRam = maxRam / 1024
    
    resultString += "path_to_goal: " + str(solution.getMoveList()) + "\n"
    resultString += "cost_of_path: " + str(len(solution.getMoveList())) + "\n"
    resultString += "nodes_expanded: " + str(len(explored)) + "\n"
    resultString += "fringe_size: " + str(len(fringe)) + "\n"
    resultString += "max_fringe_size: " + str(maxFringe) + "\n"
    resultString += "search_depth: " + str(solution.getDepth()) + "\n"
    resultString += "max_search_depth: " + str(maxDepth) + "\n"
    resultString += "running_time: " + "{0:.8f}".format(totalTime) + "\n"
    resultString += "max_ram_usage: " + "{0:.8f}".format(maxRam) + "\n"
    
    f = open('output.txt', 'w')
    f.write(resultString)
    f.close()
    return





global PrStartTime
global explored
global fringe
global maxFringe
global maxDepth
global maxRam

PrStartTime = time.time();

algorithm = sys.argv[1]
try:
    paramlist = sys.argv[2]
except IndexError:
    paramlist = algorithm

original = [int(i) for i in paramlist.split(",")]

final = list(range(len(original)))

n = int(sqrt(len(original)))

startboard = board(original)

rootNode = searchNode(startboard, []) 

if(algorithm=="bfs"):
    solution = solver.bfs(rootNode, final)
elif(algorithm=="dfs"):
    solution = solver.dfs(rootNode, final, 0)
elif(algorithm=="ast"):
    solution = solver.ast(rootNode, final)
else:
    solution = solver.ida(rootNode, final)

if(solution):
    finalizer(solution)


    
    
    
    

    
    
    
    
    
    
    
