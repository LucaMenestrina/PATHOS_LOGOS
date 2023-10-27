## LOGOS
### Learning Optimized Graph-based representations of Object Semantics

<br>

Hyperparameter Optimization can be launched with:  
```
python LOGOS_HPO.py
```
<br>

Comparison of 4 models (RotatE, ComplEx, TransE, DistMult) with 5 replicas each can be launched with:
```
python  LOGOS_compare.py
```

For performing the comparison on a cluster the file ```run.py``` is available in the ```comparison``` folder.  
It can be launched with:
```
python run.py -m "model_name" -r replica_number 
```
e.g.
```
python run.py -m "RotatE" -r 1
```

The predictions can be repeated with:
```
python LOGOS_predict.py
```