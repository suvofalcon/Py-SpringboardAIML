"""
Created on 03-Oct-2016

@author: suvosmac
"""
##########################################################################
# WorkingWithFiles.py -
# We will demonstrate how to work with files in Python
# How to read and write files
# Python version used at the time of creation - 2.7.11
##########################################################################

# We will use the below syntax for opening a text file
my_file = open("/Volumes/Data/CodeMagic/Data Files/wordcount.txt")

# verify its a file object
print(type(my_file))

# Once we have created a file object, we have to read its contents
file_contents = my_file.read()

# we can print the entire contents of the file, but this is not advisable
# for large files :)
print(file_contents)
# We can also print its length
print("Length of the contents in the file : %d" % (file_contents.__len__()))

# When we print the contents of the file, we see that Python has included a new line character at the end of every
# sentence and so every line has a sentence. This also potentially gives a wrong length information.
# To avoid this problem, we strip the new line character after we read it
try:
    my_file = open("/Volumes/Data/CodeMagic/Data Files/wordcount1.txt")
    file_contents = my_file.read().rstrip("\n")
    # Now we print the contents and the length
    print('Contents of the file : %s \n The length is %d' %
          (file_contents, file_contents.__len__()))
except IOError as error:
    print("There is no such file - %s" % error)
except AttributeError as error:
    print("Attribute Error - %s" % error)

# Writing Text to the file
# In Python to write to a file, we open the file with  mode - r (read), w(write) and a(append)
# If we do not specify anything, the default is "r(read)"
# The difference between "w" and "a" is subtle, but important. If we open a file that already exists using
# the mode "w", then we will overwrite the current contents with whatever data we write to it.
# If we open an existing file with the mode "a", it will add new data onto the end of the file, but will not
# remove any existing content. If there doesn't already exist a file with the specified name, then "w" and "a"
# behave identically – they will both create a new file to hold the output.

# Let us open a file for writing

my_file = open("/Users/SUVOSMAC/Downloads/testFile.txt", "w")
# With write we have to use a string as argument
my_file.write("Hello World from Python!!")
my_file.write(str(6))  # To write a number
my_file.close()

# Get user input
# Including a new line character so that user can type in the next line
name = input("What is your name?")
print("Hello " + name)

# We can also use the user input for files
filename = input("Give the file name..\n")
input_file = open("/Users/SUVOSMAC/Downloads/" + filename + ".txt", "w")
input_file.write("Writing")
input_file.close()