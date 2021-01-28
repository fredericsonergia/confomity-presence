def convertPoints2BndBox(points):
    xmin = float("inf")
    ymin = float("inf")
    xmax = float("-inf")
    ymax = float("-inf")
    for p in points:
        x = p[0]
        y = p[1]
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)

    if xmin < 1:
        xmin = 1

    if ymin < 1:
        ymin = 1

    return (int(xmin), int(ymin), int(xmax), int(ymax))
