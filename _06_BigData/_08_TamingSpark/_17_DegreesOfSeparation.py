'''
Represent each line as a node with connections, a color and a distance

for example
5983 1165 3836 4361 1282

becomes
(5983, (1165, 3836, 4361, 1282), 9999, WHITE)

Initial condition is that node is infinitely distant (9999) and WHITE

Go through looking for Gray nodes to expand
Color nodes we are done with Black
Update the distances as we go

The mapper
Create new nodes for each connection of gray nodes with a distance incremented by one, color gray and no connections
colors the gray node as black just being procesed
copies the node itself into the results

The reducer
combines together all nodes for the same heroID
Preserves the shortest distance and the darkest color found
Preserves the list of connections from the original node

An accumlator allows many executors to increment a shared variable
'''

# Import the Spark Libraries
from pyspark import SparkConf, SparkContext

# Initialize Spark Config and set application name
conf = SparkConf().setMaster("local").setAppName("DegreesOfSeparation")
sc = SparkContext(conf=conf)

# The characters we wished to find the degree of separation between:
startCharacterID = 5306 # Spiderman
targetCharacterID = 14 # ADAM

# Define a accumulator, used to signal when we find the target Character uding our BFS traversal
# Sets up a shared accumulator with initial value 0
hitCounter = sc.accumulator(0)

'''
Convert to BFS structure
'''
def convertToBFS(line):
    fields = line.split()
    heroID = fields[0]
    connections = []
    for items in fields[1:]:
        connections.append(int(items))
    
    color = "WHITE"
    distance = 9999

    if (heroID == startCharacterID):
        color = "GRAY"
        distance = 0
    
    return (heroID,(connections, distance, color))

'''
Read the data file and call the mapper function
'''
def createStartingRDD():
    inputFile = sc.textFile("../resources/Marvel-Graph")
    return (inputFile.map(convertToBFS))

'''
BFS Mapper
'''
def bfsMap(node):
    characterID = node[0]
    data = node[1]
    connections = data[0]
    distance = data[1]
    color = data[2]

    results = []

    # If this node needs to be expanded...
    if (color == 'GRAY'):
        for connection in connections:
            newCharacterID = connection
            newDistance = distance + 1
            newColor = "GRAY"
            if (targetCharacterID == connection):
                hitCounter.add(1)
            
            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)
        
        # We have processed this node, hence color it Black
        color = "BLACK"
    
    # Emit the input node, so we dont lose it
    results.append((characterID, (connections, distance, color)))
    return results

'''
BFS Reducer
'''
def bfsReduce(data1, data2):
    edges1 = data1[0]
    edges2 = data2[0]
    distance1 = data1[1]
    distance2 = data2[1]
    color1 = data1[2]
    color2 = data2[2]

    distance = 9999
    color = color1
    edges = []

    # See if one is the original node with its connections.
    # If so preserve them.
    if (len(edges1) > 0):
        edges.extend(edges1)
    if (len(edges2) > 0):
        edges.extend(edges2)

    # Preserve minimum distance
    if (distance1 < distance):
        distance = distance1

    if (distance2 < distance):
        distance = distance2

    # Preserve darkest color
    if (color1 == 'WHITE' and (color2 == 'GRAY' or color2 == 'BLACK')):
        color = color2

    if (color1 == 'GRAY' and color2 == 'BLACK'):
        color = color2

    if (color2 == 'WHITE' and (color1 == 'GRAY' or color1 == 'BLACK')):
        color = color1

    if (color2 == 'GRAY' and color1 == 'BLACK'):
        color = color1

    return (edges, distance, color)


'''
Main Program Begins here
'''
# One of the sample element of iterationRDD is 
# ('5632', ([2912, 4366, 2040, 1602, 4395, 133, 403, 2178, 6306], 9999, 'WHITE'))
iterationRdd = createStartingRDD()
for iteration in range(0, 10):
    print("Running BFS iteration# " + str(iteration+1))

    # Create new vertices as needed to darken or reduce distances in the
    # reduce stage. If we encounter the node we're looking for as a GRAY
    # node, increment our accumulator to signal that we're done.
    mapped = iterationRdd.flatMap(bfsMap)

    # Note that mapped.count() action here forces the RDD to be evaluated, and
    # that's the only reason our accumulator is actually updated.
    print("Processing " + str(mapped.count()) + " values.")

    if (hitCounter.value > 0):
        print("Hit the target character! From " + str(hitCounter.value) \
            + " different direction(s).")
        break

    # Reducer combines data for each character ID, preserving the darkest
    # color and shortest path.
    iterationRdd = mapped.reduceByKey(bfsReduce)



