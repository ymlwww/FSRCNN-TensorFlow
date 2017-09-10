import sys

def main():
  scale = 2
  radius = 2
  size = radius * scale * 2 + 1
  if len(sys.argv) == 2:
    fname=sys.argv[1]
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    x=list(reversed(range(scale)))
    x=x[-1:]+x[:-1]
    xy = []
    for i in x:
        for j in x:
            xy.append([j, i])

    m = []
    for i in range(0, len(xy)):
        xi, yi = xy[i]
        for y in range(yi, size, scale):
            for x in range(xi, size, scale):
                m.append(y + x * size)
    #print(m)
    content = list(reversed(content))
    with open('sorted.txt', 'w') as file:
        for l in range(0, len(m)):        
            file.write("(" + content[m[l]].strip(",") + "),\n")
  else:
    print("Missing argument: You must specify a file name that contains the deconvolution weights")
    return

if __name__ == '__main__':
  main()
