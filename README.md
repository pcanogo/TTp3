# TTp3
parallelized k-means implementation

## Dependencies
Install pip dependencies:
```
$ pip install numpy
$ pip install nltk
$ pip install pandas
```

then install nltk dependencies: 
```
$ python 
	import nltk
	nltk.download('stopwords')
	nltk.download('punkt')
```

## How to run
To run enter the following command:

```
$ sh kmeans.sh -[s|p] [p:number_of_processes]
```
Where: 
 * -s will run the serial kmeans implementation
 * -p will run the parallel kmeans implementation along with the number of specdified procecess  
