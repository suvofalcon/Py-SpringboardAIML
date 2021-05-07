"""
Created on 03-Oct-2016

@author: suvosmac
"""

##########################################################################
# Functions.py -
# Demonstrates how you use functions in Python
# Python version used at the time of creation - 3.x
##########################################################################

# Variable declarations
fridge = {"apples": 10, "oranges": 3, "milk": 2}
special_sauce = ["ketchup", "mayonnaise", "french dressing"]

# Define a function
# -----------------------------


def in_fridge(wanted_food):
    """ This is a function to see if the fridge has food. Fridge has to be a dictionary defined outside
    the function. The food to be searched for is in the string wanted_food"""
    try:
        count = fridge[wanted_food]
    except KeyError:
        count = 0
    return count


def in_fridge2(someFridge, desiredItem):
    """This is a second version of the in_fridge function. The first parameter is a dictionary of the fridge contents
    and the second parameter is the desired content which needs to be checked"""
    try:
        count = someFridge[desiredItem]
    except KeyError:
        print("The fridge has no such contents")
        count = 0
    return count


# We will now use this function
wanted_food = "apples"
count_items = in_fridge(wanted_food)
# we will invoke the function
print("Number of Items fridge has for %s is %d" % (wanted_food, count_items))
wanted_food = "lichi"
count_items = in_fridge(wanted_food)
print("Number of Items fridge has for %s is %d" % (wanted_food, count_items))
# to display the function description
print("Function Description : %s" % in_fridge.__doc__)

# Let us use the second version of the function
print("Number of items the fridge has is %d" %
      in_fridge2({"Cookies": 7, "Broccoli": 3, "Milk": 2}, "Cookies"))

# Variable Scope discussion
# ---------------------------------
# Any name in the top - level scope can be reused in a lower - level scope without affecting the data referred
# to by the top - level name


def make_new_sauce():
    """This function makes a new special sauce all of its own..."""
    special_sauce = ["mustard", "yogurt"]
    return special_sauce


# we will refer at Global Scope
print("Special Sauce is now : %s" % special_sauce)
new_sauce = make_new_sauce()
print("Now the Special Sauce is : %s" % new_sauce)
# hence we can see when executing the function, it prints the variable contents at the local scope (function scope)
