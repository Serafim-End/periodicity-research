#### periodicity-research
The research of accuracy of searching cycles methods in symbolic sequences (time series) with presence of random noises.

#### Official topic of the work:
## Research of Accuracy Cycles Identification Methods in Symbolic Sequences with Random Noises

#### Pugacheva, Korotkov algorithm
###### main proces of the algorithm:
- based on mix of genetic algorithm and dynamic programming 
- works perfectly for small sizes of the periods
- works normally (if you wanna get more concrete digits - ask about) with huge sequences 
- works with all types of noise
- works with all types of periodicity

About all future analysis - percent (доля) in this case one 
(percent of noisy periods) * (percent of noise in period) where
percent of noisy periods = percent of noisy periods in my tests
###### Brief analysis of the picture
- just to show that algorithm (in my research) works more efficiently with small periods
 
![Korotkov](http://i63.tinypic.com/2vdihsl.png)


###### main conces of the algorithm:
- sometimes you can wait for hours to get result (if you use cluster)
- not easy to prepare the customizations for genetic algorithm 
- not easy to make good first initialization of the organisms - matrices

Examples of Data files that was tested [on cluster](http://victoria.biengi.ac.ru/)
- examples maybe here 
If you wanna test Korotkov algortihms - see the cluster link

#### CONV by Mohamed G. Elfeky, Walid G. Aref, Senior Member, IEEE, and Ahmed K. Elmagarmid
###### main proses of the algorithm:
- based on convolution
- fast computational complexity due to FFT in the kernel 
- works well with any sizes of alphabet and any sizes of sequence

###### Brief analysis of the picture
- just to show the efficiency (time complexity) of the CONV algorithm 
![time](http://i65.tinypic.com/30c84cy.png)

###### main disadvantages of the algorithm:
- does not work with any noises (works well only with ``replace`` noise)
- there are no single implementation for all types of periodicity
 (exist separated implementations for ``partial`` and ``segment`` periodicity)

###### to test this algorithm see the "instruction to test algorithms"
- additional implementation with GUI on java

#### WARP by Mohamed G. Elfeky, Walid G. Aref, Ahmed K. Elmagarmid
###### main proses of the algorithm:
- based on dynamic time warping 
- easy to understand, easy to implement

###### main cons of the algorithm:
- does not work great with all types of noises

to test this algorithm see the "instruction to test algorithms"
- additional implementation with GUI on java

#### STNR by Faraz Rasheed, Reda Alhajj
###### main feature of the algorithm:
- based on suffix tree 
- works great for all types of periodicities
- works well for all types of noise

###### main cons of the algorithm:
- works well only for discrete small alphabet

to test this algorithm ask about additional implementation that based on masudio solution


###### instruction to test algorithms: 
- generate_data.py - customize and run the file
- test.py choose the test function and run it

_______________________________________________________________________________

Send me email to endnikita@gmail.com if you want to compare these algorithms with any simple like algorithms below 
All of them were implemented on Java with GUI interface (JavaFX)

##### Available algorithms to compare with:
#### Floyd algorithm
###### pros: 
- easy to understand, easy to implement
###### cons:
- no chances to detect difficult periodicity (more than one same elements in a period)
- single cycle estimation

#### Brean algorithm
###### pros and cons are the same 
- modification of Floyd (binary degree representation) that estimate period more correct

#### Gosper algorithm
######pros and cons are the same as Floyd
- independent implementation 

#### Nivash algorithm
######pros and cons:
- very fast 
- more correct than previous 
- was developed only in 2004
