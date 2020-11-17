#!/opt/anaconda3/bin/python
def testmethod(): 
    print("Testing OK")

class MyClass:

    def __init__(self):
        print("Initializing MyClass")
    myvar = "blah"
    __privateVariable = "Private"
    newVar = 'that'

    def myfun(self):
        print("This is a message inside the class.")
    def unreachableFunc():
        print("what?!?!")
        print(self.myvar)

    def __str__(self):
        return self.newVar + " " + self.myvar
        
