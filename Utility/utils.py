import numpy as np
import matplotlib.pyplot as plt
import os

DEBUG = False

def cell_variance(features1, features2, img_size, levels=3):

    score=0
    for i in range(levels):
        # Number of cells along a given axis
        res = 2**(i+1)

        # The width of those cells along each axis
        increment = [int(np.ceil(x/res)) for x in img_size]

        # Create empty array for the grid cells, likely to be sparse
        cells1 = np.zeros(res**2,dtype=int)
        cells2 = np.zeros(res**2,dtype=int)

        # Set each grid cell that is occupied to the weight of the resolution
        for f in features1:
            cells1[f[0]//increment[0] + res*(f[1]//increment[1])] = 1+i

        for f in features2:
            cells2[f[0]//increment[0] + res*(f[1]//increment[1])] = 1+i


        cells = cells1 ^ cells2

        # Visualize if in debug mode
        if DEBUG:
            ax = plt.subplot(1,levels,i+1)
            ax.set_aspect(1)
            extent = [0,img_size[0],img_size[1],0]
            plt.imshow(np.array(cells).reshape(res,-1), extent=extent, cmap='gray')
            ax.set_xticks(np.linspace(0,img_size[0],res+1))
            ax.set_yticks(np.linspace(0,img_size[1],res+1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid()
            plt.xlim(0,img_size[0])
            plt.ylim(img_size[1],0)
            plt.plot([x[0] for x in features1], [x[1] for x in features1],'yo')
            plt.plot([x[0] for x in features2], [x[1] for x in features2],'rx')
            plt.title(f"Level {i+1}")

        # Since entries are proportional to the resolution each score is already weighted
        score+=sum(cells)

    if DEBUG:
        plt.suptitle(f"Score: {score}")
        plt.show()

    return score

# def img_renamer(path):
#     for f in os.listdir(path):
        