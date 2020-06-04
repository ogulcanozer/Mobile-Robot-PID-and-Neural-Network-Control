#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Boilerplate.
#-------------------------------------------------------------------------------
from __future__ import division
import  sys, numpy , matplotlib.pyplot as plt

def histogram (im):
    "Determine the histogram of an image -- simple version."
    global MAXGREY
    myInput = []
    with open(im) as f: 
        myInput=[(line.rstrip()).split(',') for line in f]
            
    myInput = numpy.array(myInput).astype(numpy.float)
    print("--------------------------------------------------------------------------------")
    print(myInput[0].min())
    print(myInput[0].max())       
    print(myInput[1].min())
    print(myInput[1].max())
    print("--------------------------------------------------------------------------------")
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(myInput[0], bins=100)
    axs[1].hist(myInput[1], bins=100)    
    plt.show()

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------
if len (sys.argv) < 2:
    print >>sys.stderr, "Usage:", sys.argv[0], "<image>..."
    sys.exit (1)

# Process each file on the command line in turn.
for fn in sys.argv[1:]:
    histogram(fn)
    
#-------------------------------------------------------------------------------
# End of summarize.
#-------------------------------------------------------------------------------
