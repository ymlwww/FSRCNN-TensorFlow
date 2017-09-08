import sys

def main():
  scale = 2
  radius = [3, 5, 7][scale-2]
  if len(sys.argv) == 2:
    fname=sys.argv[1]
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    xy = []
    for i in range(0, scale):
        for j in range(0, scale):
            xy.append([j, i])
    xy = list(reversed(xy))

    m = []
    for i in range(0, len(xy)):
        xi, yi = xy[i]
        for x in range(xi, radius*2+1, scale):
            for y in range(yi, radius*2+1, scale):
                m.append(y + x*(radius*2+1))
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
