## Food Vector Sorting
### Introduction
* food vector: 
	- \<taste, health\> (taste: [1, 10], health:[1, 10])
* Requirement:
	* middle left of the sequence
		- the left one's taste value must be bigger than the right one
		- the left one's health value would better be smaller than the right one
	* middle right of the sequence
		- the right one's health value must be bigger than the left one
		- the right one's taste value would better be smaller than the left one
### Sorting Method
- By Linear Optimization (2017/10/3)
- By Generic Algorithm for Sorting
	* Fitness function
	* Cross-over method
		* Every vector should be included in the sequence *(refering to TSP problem solved by generic algorithm)*