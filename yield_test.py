def gen(n):
    for i in range(n):
        yield i

for i in gen(5):
    print(i)