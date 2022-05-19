class Infix:

    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    # def __rlshift__(self, other):
    #     return Infix(lambda x, self=self, other=other: self.function(other, x))
    # def __rshift__(self, other):
    #     return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)


x = Infix(lambda x, y: x**y)

print(12/12 + 454 + 95 - 156 * 2**5 * (23 + 12/1 + 5) + 23*5)     #asfd
print('Hello world')     #safsdjf
a = 1|x|2
