"""
Created on 03-Oct-2016

@author: suvosmac
"""

##########################################################################
# Decisions.py -
# how Python makes decisions
# Various Loops
# Fundamental approach for exception handling
# Python version used at the time of creation - 2.7.11
##########################################################################

# Testing for Equality
# ------------------------
print(1 == 1)
print(1 == 2)
# When we have different types of numbers, python would still be able to
# compare them
print(1.0 == 1)
# We can also test, whether the strings have the same contents
str1 = "Blackberry"
str2 = "Blueberry"
print(str1 == str2)
print((len(str1) - 1) == len(str2))
# Python considers two sequences to be equal when every element is equal
# and is in the same position in each list
apples = ["Mackintosh", "Golden Delicious", "Fuji", "Mitsu"]
apple_trees = ["Golden Delicious", "Mitsu", "Fuji", "Mackintosh"]
print(apples == apple_trees)
apple_trees = ["Mackintosh", "Golden Delicious", "Fuji", "Mitsu"]
print(apples == apple_trees)
# Like list as before, Dictionaries can also be compared and they are equal only if both the keys and the values in
# respective positions are equal
tuesday_breakfast_sold = {
    "pancakes": 10, "french toast": 4, "bagels": 32, "eggs and sausages": 13}
wednesday_breakfast_sold = {
    "pancakes": 15, "french toast": 4, "bagels": 33, "eggs and sausages": 10}
thursday_breakfast_sold = {
    "pancakes": 10, "french toast": 4, "bagels": 32, "eggs and sausages": 13}
print(tuesday_breakfast_sold == wednesday_breakfast_sold)
print(tuesday_breakfast_sold == thursday_breakfast_sold)

# Doing the Opposite - Not equal
# -----------------------------
print(3 != 3)
print(5 != 4)
# also holds true for more complex types like sequences and dictionaries
print(tuesday_breakfast_sold != wednesday_breakfast_sold)
print(tuesday_breakfast_sold != thursday_breakfast_sold)

# Comparing values - Which one is more?
# -----------------------------------
print(5 < 3)
print(10 > 2)
print(10 >= 9)
# For comparing strings Python follows the below convention
# A < B < C < .................a < b < c .............< z
# Python starts doing the comparison from the first letter, if those are
# same, the next letter is compared
print("Zebra" > "aardvark")
print("Zebra" < "Zebrb")
# If the string is same but differs in the case, we can convert everything
# to lower or upper case and compare
print("Lower" == "lower")
print("Lower".lower() == "lower")

# Reversing True and False
# -----------------------------------
# not will return reverse of the outcome of the test
print(not 5 > 2)
# print(not "A" < 3) # throws an error with Python 3.x onwards

# How to get decisions made (If Loop)
# -----------------------------------
# Demos a nested if condition
omlet_ingredients = {
    "egg": 2, "mushroom": 5, "pepper": 1, "cheese": 1, "milk": 1}
fridge_contents = {"egg": 10, "mushroom": 20,
                   "pepper": 3, "cheese": 2, "tomato": 4, "milk": 15}
have_ingredients = [False]
if fridge_contents["egg"] >= omlet_ingredients["egg"]:
    have_ingredients[0] = True
have_ingredients.append("egg")
print(have_ingredients)

if fridge_contents["mushroom"] >= omlet_ingredients["mushroom"]:
    if have_ingredients[0] == False:
        have_ingredients[0] = True
    have_ingredients.append("mushroom")
print(have_ingredients)

# Demo of how to handle errors
# -----------------------------------
# we see the tomato ingredient is not there
try:
    if omlet_ingredients["tomato"] >= 4:
        print("Tomato is there")
except KeyError as error:
    print("There is no %s" % error)
# We want to handle this error but not want to do anything with it
except TypeError:
    pass

# Demo If-else-if
# -----------------------------------
milk_price = 2.5
if milk_price < 1.25:
    print("Buy Two cartons of milk, they are on sale!!")
elif milk_price < 2.0:
    print("Buy One carton of milk, prices are normal.")
elif milk_price > 2.0:
    print("Milk prices are sky rocketing, drink water... ")
else:
    # Also called fall through statement
    print("Never Mind, Milk is not needed")

# While Loop
# ------------------------------
index = 10
while index > 0:
    print("Launch in : %d seconds" % index)
    index = index - 1

# For Loop
# ------------------------------
for index in range(10, 0, -1):
    print("T-minus : %d" % index)