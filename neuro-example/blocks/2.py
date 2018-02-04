value = 0b101111101
mask = 0
deg = 1
for i in xrange(3):
    mask += deg
    deg *= 2

print(mask)

print (value & mask)
