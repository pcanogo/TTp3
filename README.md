# TTp3
parallelized k-means implementation

## Dependencies
Install pip dependencies:
```
$ pip install numpy
$ pip install nltk
$ pip install pandas
$ pip install mpi4py
```

then install nltk dependencies: 
```
$ python 
	import nltk
	nltk.download('stopwords')
	nltk.download('punkt')
```

or create Conda environment with 

Install pip dependencies:
```
$ conda create -n env_name python=2.7 anaconda
$ source activate env_name
```

install nltk dependencies as describe above and after install mpi4py

```
$ pip install mpi4pi
```


## How to run
To run enter the following command:

```
$ sh kmeans.sh -[s|p] [p:number_of_processes]
```
Where: 
 * -s will run the serial kmeans implementation
 * -p will run the parallel kmeans implementation along with the number of specdified procecess  
