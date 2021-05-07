"""
Created on 03-Oct-2016
@author: suvosmac
"""

##########################################################################
# _01_LanguageBasics.py -
# This code block introduces the bare basic building blocks of Python
#  - String Operations
#  - Numbers and Operators
#  - Variables and Data Types
# Python version used at the time of creation - 2.7.11
##########################################################################

# The print() function is used to print text to the screen
print("Hello World!!")

# Understanding different quotes
# ------------------------------------------------------------------------------------------------------
print('This is a string using a single quote!')
print("This is a string using a double quote!")
print("""This string has three quotes!
        Look at what it can do!!""")
# Triple quote is useful, when we want to print large amount of data and cannot be print in one line...
# Single quotes and double quote is interchangeable, except when we want
# to work with a contraction like Don't
print("I said, Don't do it...")
# to print in multiple lines, we use a new line character
print("Hello\nWorld")

# Using print function to join distinct strings together
# ------------------------------------------------------------------------------------------------------
print("John" + " Everyman")  # OR
print("John" + " " + "Everyman")

# Some formatting function
# ------------------------------------------------------------------------------------------------------
print("Art : %5d, Price per Unit: %8.2f" % (453, 59.058))
# 5d means a provision of 5 digits
print("John Q. %s" % 'Public')
print("My Name is %s %s" % ("John", "Everyman"))

# Changing Case and Replacement
# -------------------------------------------------------------
my_name = "MARTIN"
print("Name is %s" % my_name)
# now we will run the lower method to change case
my_name = my_name.lower()
print("Name is %s" % my_name)

# Now we will see replacement
greeting = "Hi, friend"
# we will replace i with o
print(greeting.replace("i", "o"))
# We can also replace a word
print(greeting.replace("Hi", "Howdy"))

# String extraction
# ---------------------------------------------------------
word = "spontaneous"
# print letters three to five
print(word[3:5])  # position start at 0 and not 1
print(word[0:7])
print(word[5:])  # starts from 5 goes till end

# Counting and finding Substrings
# -------------------------------------------
word = "spontaneous"

# count some different letters
n_count = word.count('n')
ne_count = word.count('ne')
x_count = word.count('x')

# Now we will print the outputs
print("Number of n's : " + str(n_count))
print("Number of ne's : " + str(ne_count))
print("Number of x's : " + str(x_count))

# Finding the string location
# -------------------------------------------------
# In python we start counting from 0
word = "spontaneous"
print(str(word.find('n')))  # returns the first occurence of the character
print(str(word.find('ne')))
print(str(word.find('x')))  # returns -1 which means non existent

# Python has three numbers set - Integers (Positive & Negative), Float (Positive & Negative) and Imaginary
# ------------------------------------------------------------------------------------------------------
print(type(45))
print(type(-23.93))
print(type(3j + 5))  # imaginary number will have a j in it

# We will use number formats
print("$%.02f" % 30.0)
print("$%.03f" % 30.16877)

# We can also convert to an integer (provided it is a valid literal)
number = 3 + int('4')
print(number)

# Lets use variables in print formats
# ------------------------------------------------------------------------------------------------------
first_string = "This is a string"
second_string = "This is a second string"
first_number = 4
second_number = 5
result = "The first variables are %s, %s, %d, %d" % (
    first_string, second_string, first_number, second_number)
print(result)

# Changing data through variable names
proverb = "A penny saved"
proverb = proverb + " is a penny earned"
print(proverb)

# Custom Data Types
# ------------------------------------------------------------------------------------------------------
# Python provides four other important basic types: tuples, lists, sets, and dictionaries.
# They allows us to group more than one item of data together under one name.
# Each gives capability to search through them, because of the grouping

# Lets create a tuple
# ------------------------------------------------------------------------------------------------------
aTuple = ("first", "second", "third")
print("The first element of the tuple is : %s" % aTuple[0])
# To print the length of the tuple
print("Length of aTuple : %d" % len(aTuple))

# To create a tuple with one element, you have to follow that one element
# with a comma
single_element_tuple = ("the sole element",)
# A tuple can have any kind of data in it, but after you have created one it can't be changed. It is immutable, and in
# Python this is true for a few types (for instance, strings are immutable after they are created; and operations on
# them that look like they change them actually create new strings).

# Another data type in Python is a list.. to create a list we follow the below syntax
# ------------------------------------------------------------------------------------------------------
breakfast = ["coffee", "tea", "toast", "egg"]
# to access an element in the list, we go by the same approach as like a tuple
print("Today's breakfast menu is %s" % breakfast[2])
# The primary difference between list and tuple is, a list can be modified
# after it has been created
breakfast.append("waffles")
print(breakfast)
# to add multiple items at once
breakfast.extend(["juice", "decaf", "oatmeal"])
print(breakfast)
# we can also use the length function on the new list
print("The lenght of the extended list is now - %d" % len(breakfast))

# Dictionary
# ------------------------------------------------------------------------------------------------------
# A dictionary is similar to lists and tuples, its another type of container for a group of data.
# However unlike tuples and list, elements of a dictionary are not referenced by their numeric order but are indexed by
# names which can be letters or numbers
# A dictionary is created as
menus_specials = {}
menus_specials["breakfast"] = "Candadian Ham"
menus_specials["lunch"] = "tuna surprise"
menus_specials["dinner"] = "Cheeseburger Deluxe"
# A dictionary can also be created as
menus_special = {"breakfast": "Sausage and Eggs", "Lunch": "Split pea soup and garlic bread", "dinner":
                 "2 hot dogs and onion rings"}
print(menus_special)
# Referencing elements of a dictionary by name
print("%s" % menus_special["breakfast"])
# The keys of the dictionary can be retreived as
courses = menus_special.keys()
print(courses)
print(type(courses))  # This returns a list
# Specific item in a dictionary can also be retreived by
print(menus_special.get("Lunch"))

# Testing a String like a list
# Python has a feature of treating string as though it is a list of
# individual characters
last_names = ["Douglass", "Jefferson", "Williams", "Frank", "Thomas"]
print("Second name : %s" % last_names[1])
print("Third Name : %s, First Character of the last name : %s" %
      (last_names[2], last_names[2][0]))
print("%s" % last_names[len(last_names) - 2])
# Python also allows to reference the last element in a list by using
# negative number as index
print(last_names[-1])

# Lets do some slicing Sequences - We are doing it on a tuple, in the same
# manner we can do it on a list
slice_me = ("The", "next", "time", "we", "meet", "drinks", "are", "on", "me")
sliced_tuple = slice_me[5:9]
print(sliced_tuple)

# Using a tuple to form / extend a list
living_room = ("rug", "table", "chair", "TV", "dustbin", "shelf")
apartment = []
apartment.extend(living_room)
print(apartment)
# OR if we use append - we would get entire different thing
apartment = []
apartment.append(living_room)
# This will consider the tuple to be the first element of the list
print(apartment)
print(len(apartment))  # The length will be 1, because there is just one tuple

# Popping elements from a list
# lets create a list
todays_temperature = [23, 32, 33, 31]
todays_temperature.append(29)
print("This morning temperature is %.02f" % todays_temperature.pop(0))
# Once you use pop, the element goes out from the list and second element becomes the first
# In this way, we can use a list as a temporary storage
print(todays_temperature)
# If we dont tell pop which item to remove, it will by default remove the
# last item

# Sets
# ------------------------------------------------------------------------------------------------------
# We would now work with Sets
# Sets are similar to dictionaries, the only difference is they consist of only keys with no associated values
# essentially they are a collection of data with no duplicates
# Below we can assign some values and remove duplicates by assigning them
# to a set
alphabet = ["a", "b", "b", "c", "a", "d", "e"]
print(alphabet)
alpha2 = set(alphabet)
print(alpha2)
# Check the type
print(type(alpha2))

# Some Problems
# -------------------------------------------

# What proportion of this long paragraph is made up of vowels..

tale_two_cities = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of " \
                  "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of "\
                  "Light, it was the season of Darkness it was the spring of hope, it was the winter of despair, " \
                  "we had everything before us, we had nothing before us, we were all going direct to Heaven, we were "\
                  "all going direct the other way - in short, the period was so far like the present period, that "\ 
                  "some of its noisiest authorities insisted on its being received, for good or for evil, in the "\
                  "superlative degree of comparison only."

# start counting the vowels
a_count = tale_two_cities.count('a')
e_count = tale_two_cities.count('e')
i_count = tale_two_cities.count('i')
o_count = tale_two_cities.count('o')
u_count = tale_two_cities.count('u')
# find total length of the paragraph
para_len = tale_two_cities.__len__()
vowel_proportion = float(a_count + e_count + i_count + o_count + u_count) / para_len
# Proportion of vowels in the para
print("Proportion of the vowels : %0.3f" % vowel_proportion)

# Swapping letters - replace a with e and e with a

str_para = "emong othar public buildings in e cartein town, which for meny raesons it will ba prudant to rafrein " \
           "from mantioning, end to which i will essign no fictitious nema, thara is ona enciantly common to most " \
           "towns, graet or smell: to wit, e workhousa; end in this workhousa wes born; on e dey end deta which " \
           "i naad not troubla mysalf to rapaet, inesmuch es it cen ba of no possibla consaquanca to tha raedar, " \
           "in this stega of tha businass et ell avants; tha itam of mortelity whosa nema is prafixad to tha haed of " \
           "this cheptar. "

# we will use a temporary variable for the replacement...
# first we will replace all 'a' with '*' and then replace all e with a..
# and then replace all '*' with 'e'
str_para = str_para.replace("a", "*")
str_para = str_para.replace("e", "a")  # and finally
str_para = str_para.replace("*", "a")
# check for correctness
print(str_para)

# Count the length of the Sentences
str_para = "Night is generally my time for walking. In the summer I often leave home early in the morning, " \
           "and roam about fields and lanes all day, or even escape for days or weeks together; but, saving in the " \
           "country, I seldom go out until after dark, though, Heaven be thanked, I love its light and feel the "\
           "cheerfulness it sheds upon the earth, as much as any creature living."

index = 1
while len(str_para) > 0:
    # extract the sentence
    sentence_location = str_para.find('.')
    sentence = str_para[0:sentence_location]
    print("Length of the sentence %d is %d" % (index, len(sentence)))
    str_para = str_para[sentence_location + 1:]
    index = index + 1