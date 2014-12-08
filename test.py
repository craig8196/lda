


d = {
    'a': 23,
    'b': 25,
    'c': 45,
    'f': 1,
}

print max(d, key=lambda x: d[x])

def key(x):
    return d[x]

def cmp(a,b):
    return a-b

print sorted(d, cmp=lambda x,y: x-y, key=lambda x: d[x])
