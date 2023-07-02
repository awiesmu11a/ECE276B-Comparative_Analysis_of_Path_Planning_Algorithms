# Implementation ideas:

## Baseline implementation:

* Firstly generate voxels: not efficient for simple astar
* Mandatory for any kind of operation since the given  space is continuous so convert to discrete
* use variant of Bresenham's algorithm to check the collision
* take the nodes from and pass to a function along with boundary, boxes, resolution, and two nodes; if along the line any of the voxels in the obstacle class collision takes place
* Yet to decide on resolution
* Will depend on the thickness of the boxes.
* Now implementing baseline A* and RRT implementation not a big task

## Additional:

* Upgrade both sampling and search based simultaneously by one step at a time for consistency in comparison
* Thinking multi resolution to speed up algorithm (fine near obstacles and coarse away from them); Also look for RRT*
* Will be reading papers on A* variants try to implement (will be helpful for Heirarchical Motion Planning)
