# Grassfire transform

The [Grassfire transform](https://en.wikipedia.org/wiki/Grassfire_transform) is a simple algorithm is a pixel-discrete algorithm to extract the skeleton or medial axis of a region.

`cv_algorithms` provides an easy-to-use implementation for Python & OpenCV. Due to it being implemented in C, it is suitable for high-performance applications.

For a simple example source code see [this file](https://github.com/ulikoehler/cv_algorithms/blob/master/examples/grassfire.py).

## Result example

This example has been generated using the example script linked to above.

Input:

![Grassfire input](https://raw.githubusercontent.com/ulikoehler/cv_algorithms/master/examples/grassfire-example.png)

Output:

![Grassfire result](https://raw.githubusercontent.com/ulikoehler/cv_algorithms/master/examples/grassfire-result.png)


## A pure python implementation

In order to better understand the algorithm, you can have a look at this pure Python version:

```python

def grassfire_transform(mask):
    """
    Apply the grassfire transform to a binary mask array.
    """
    h, w = mask.shape
    # Use uint32 to avoid overflow
    grassfire = np.zeros_like(mask, dtype=np.uint32)

    # 1st pass
    # Left to right, top to bottom
    for x in range(w):
        for y in range(h):
            if imgGray[y, x] != 0: # Pixel in contour
                north = 0 if y == 0 else grassfire[y - 1, x]
                west = 0 if x == 0 else grassfire[y, x - 1]
                if x == 3 and y == 3:
                    print(north, west)
                grassfire[y, x] = 1 + min(west, north)

    # 2nd pass
    # Right to left, bottom to top
    for x in range(w - 1, -1, -1):
        for y in range(h - 1, -1, -1):
            if gf[y, x] != 0: # Pixel in contour
                south = 0 if y == (h - 1) else grassfire[y + 1, x]
                east = 0 if x == (w - 1) else grassfire[y, x + 1]
                grassfire[y, x] = min(grassfire[y, x],
                    1 + min(south, east))
    return grassfire

```
